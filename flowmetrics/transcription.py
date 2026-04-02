"""
Módulo de transcripción y diarización.

Pipeline con cadena de fallback:
1. WhisperX + Sortformer v2 + TitaNet (mejor calidad, requiere NeMo/WSL2)
2. WhisperX + pyannote (buena calidad, requiere token HF)
3. Whisper estándar + diarización espectral MFCC+KMeans (funciona siempre)
"""

import os
import sys
import re
import logging
import numpy as np
import librosa

# Suprimir logs [NeMo W/I ...] antes de cualquier import de NeMo
for _logger_name in ['nemo_logger', 'nemo', 'pytorch_lightning', 'lightning.fabric', 'lightning.pytorch']:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

from .config import (
    SR, WHISPERX_MODEL, WHISPER_FALLBACK_MODEL, COMPUTE_TYPE,
    HF_TOKEN, SORTFORMER_MODEL_NAME, TITANET_MODEL_NAME,
    BACKEND_SORTFORMER, BACKEND_PYANNOTE,
)


# ============================================================================
# TRANSCRIPCIÓN POR CHUNKS (evita que el VAD se pierda en pausas largas)
# ============================================================================

def _chunked_transcribe(model, audio, chunk_s=90, overlap_s=15):
    """Transcribe audio largo en ventanas solapadas.

    Cuando hay pausas largas con ruido de público (9-10s de gritos),
    el VAD de WhisperX "se rinde". Partir en chunks solapados evita esto:
    cada chunk tiene su propio pase de VAD desde cero.

    Deduplicación: en zonas de overlap, si dos segmentos tienen start
    dentro de 1.5s, se queda el de mayor contenido.
    """
    sr = 16000
    total_samples = len(audio)
    total_duration = total_samples / sr

    if total_duration <= chunk_s + 10:
        result = model.transcribe(audio, batch_size=16, language="es")
        return result.get("segments", [])

    chunk_samples = int(chunk_s * sr)
    overlap_samples = int(overlap_s * sr)
    step_samples = chunk_samples - overlap_samples

    all_segments = []
    chunk_idx = 0
    pos = 0

    print(f"  📦 Chunks de {chunk_s}s ({total_duration:.0f}s de audio)")

    while pos < total_samples:
        end = min(pos + chunk_samples, total_samples)
        chunk = audio[pos:end]
        chunk_start_s = pos / sr
        chunk_end_s = end / sr
        chunk_dur = (end - pos) / sr

        if chunk_dur < 5:
            break

        print(f"    Chunk {chunk_idx}: {chunk_start_s:.0f}s → {chunk_end_s:.0f}s")
        result = model.transcribe(chunk, batch_size=16, language="es")

        for seg in result.get("segments", []):
            seg["start"] = seg.get("start", 0) + chunk_start_s
            seg["end"] = seg.get("end", 0) + chunk_start_s
            all_segments.append(seg)

        chunk_idx += 1
        pos += step_samples

    # Deduplicar
    if all_segments:
        all_segments.sort(key=lambda s: s.get("start", 0))
        deduped = [all_segments[0]]
        for seg in all_segments[1:]:
            prev = deduped[-1]
            time_diff = abs(seg.get("start", 0) - prev.get("start", 0))
            if time_diff < 1.5:
                if len(seg.get("text", "")) > len(prev.get("text", "")):
                    deduped[-1] = seg
            else:
                deduped.append(seg)
        n_dupes = len(all_segments) - len(deduped)
        if n_dupes > 0:
            print(f"    Dedup: {n_dupes} duplicados eliminados")
        all_segments = deduped

    print(f"  ✅ {len(all_segments)} segmentos de {chunk_idx} chunks")
    return all_segments


# ============================================================================
# DIARIZACIÓN NVIDIA (SORTFORMER v2 + TITANET)
# ============================================================================

def _extract_titanet_embeddings(audio_path, diarize_df, device):
    """Extrae embeddings de TitaNet (192 dims) para cada speaker.

    TitaNet genera un vector compacto que captura la identidad vocal.
    Para cada speaker: concatena segmentos (máx 30s), escribe wav
    temporal y extrae el vector.

    Returns:
        dict {speaker_label: list[float]} con embeddings de 192 dims
    """
    import torch
    import soundfile as sf
    import tempfile
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    print(f"  ⏳ Extrayendo embeddings TitaNet...")
    titanet = EncDecSpeakerLabelModel.from_pretrained(TITANET_MODEL_NAME)
    titanet = titanet.to(device)
    titanet.eval()

    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    embeddings = {}
    speakers = diarize_df["speaker"].unique()

    for spk in speakers:
        spk_segs = diarize_df[diarize_df["speaker"] == spk]
        chunks = []
        total_dur = 0.0

        for _, row in spk_segs.iterrows():
            start_sample = int(row["start"] * sr)
            end_sample = int(row["end"] * sr)
            chunk = y[start_sample:end_sample]
            chunk_dur = len(chunk) / sr
            if total_dur + chunk_dur > 30.0:
                remaining = 30.0 - total_dur
                chunk = chunk[:int(remaining * sr)]
                chunks.append(chunk)
                break
            chunks.append(chunk)
            total_dur += chunk_dur

        if not chunks:
            continue

        concatenated = np.concatenate(chunks)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, concatenated, sr)

            with torch.no_grad():
                audio_tensor = torch.tensor(
                    concatenated, dtype=torch.float32,
                ).unsqueeze(0).to(device)
                audio_len = torch.tensor(
                    [len(concatenated)], dtype=torch.long,
                ).to(device)
                logits, emb_tensor = titanet.forward(
                    input_signal=audio_tensor,
                    input_signal_length=audio_len,
                )
                emb_np = emb_tensor.squeeze().cpu().numpy()
                embeddings[spk] = emb_np.tolist()
                print(f"    ✅ {spk}: {len(embeddings[spk])} dims")
        except Exception as e:
            print(f"    ⚠️  Error embedding {spk}: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return embeddings


def _try_diarize_sortformer(audio_path, min_speakers=2):
    """Diarización con NVIDIA Sortformer v2 (DER ~6.57%).

    Sortformer v2 es end-to-end — detecta automáticamente cuántos
    speakers hay (hasta 4) sin clustering externo.

    Returns:
        (diarize_df, embeddings) o (None, {}) si falla
    """
    if not BACKEND_SORTFORMER:
        return None, {}

    tmp_mono_path = None
    try:
        import torch
        import pandas as pd
        import soundfile as sf
        import tempfile
        from nemo.collections.asr.models import SortformerEncLabelModel

        device = "cuda" if torch.cuda.is_available() else "cpu"

        info = sf.info(audio_path)
        if info.channels > 1:
            print(f"  🔄 Estéreo → mono...")
            y_raw, sr_raw = librosa.load(audio_path, sr=None, mono=True)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_mono_path = tmp.name
            tmp.close()
            sf.write(tmp_mono_path, y_raw, sr_raw)
            diarize_path = tmp_mono_path
        else:
            diarize_path = audio_path

        print(f"  ⏳ Sortformer v2 en {device}...")
        sortformer = SortformerEncLabelModel.from_pretrained(SORTFORMER_MODEL_NAME)
        sortformer = sortformer.to(device)
        sortformer.eval()

        annotations = sortformer.diarize(
            audio=[diarize_path], batch_size=1, num_workers=0,
        )

        rttm_pattern = re.compile(
            r"(\d+\.?\d*)\s+(\d+\.?\d*)\s+(speaker_\d+)", re.IGNORECASE,
        )

        rows = []
        if annotations and len(annotations) > 0:
            for seg in annotations[0]:
                if isinstance(seg, str):
                    match = rttm_pattern.match(seg.strip())
                    if match:
                        start, end, speaker = match.groups()
                        rows.append({"start": float(start), "end": float(end),
                                     "speaker": speaker})
                elif isinstance(seg, tuple) and len(seg) >= 3:
                    rows.append({"start": float(seg[0]), "end": float(seg[1]),
                                 "speaker": str(seg[2])})
                elif hasattr(seg, 'start'):
                    rows.append({
                        "start": float(seg.start), "end": float(seg.end),
                        "speaker": str(getattr(seg, 'speaker',
                                               getattr(seg, 'label', 'unknown'))),
                    })

        if not rows:
            print("  ⚠️  Sortformer no detectó segmentos")
            return None, {}

        diarize_df = pd.DataFrame(rows)
        unique_speakers = sorted(diarize_df["speaker"].unique())
        label_map = {old: f"SPEAKER_{i:02d}" for i, old in enumerate(unique_speakers)}
        diarize_df["speaker"] = diarize_df["speaker"].map(label_map)

        print(f"  ✅ Sortformer: {len(unique_speakers)} speakers, {len(rows)} segmentos")
        embeddings = _extract_titanet_embeddings(audio_path, diarize_df, device)
        return diarize_df, embeddings

    except Exception as e:
        print(f"  ⚠️  Sortformer falló: {e}")
        import traceback
        traceback.print_exc()
        return None, {}
    finally:
        if tmp_mono_path and os.path.exists(tmp_mono_path):
            os.unlink(tmp_mono_path)


# ============================================================================
# PIPELINE PRINCIPAL DE TRANSCRIPCIÓN
# ============================================================================

def transcribe_whisperx(path, min_speakers=2, max_speakers=None):
    """Transcripción con WhisperX + diarización con cadena de fallback.

    Args:
        path: ruta al archivo de audio
        min_speakers: mínimo de speakers esperados (default 2 para batallas)
        max_speakers: máximo de speakers. None = dejar que pyannote decida.
            Para batallas con host, usar 4-5 para no forzar merge de voces.

    Returns:
        (segments, embeddings, emb_type) donde emb_type es:
        "titanet" | "pyannote" | "whisperx_no_diarize" | "failed"
    """
    print(f"\n📝 [PASO 5] WhisperX...")
    try:
        import torch
        import whisperx
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"  ⏳ Modelo {WHISPERX_MODEL} en {device}...")
        model = whisperx.load_model(
            WHISPERX_MODEL, device,
            compute_type=COMPUTE_TYPE, language="es",
        )
        audio = whisperx.load_audio(path)

        print(f"  ⏳ Transcribiendo...")
        segments = _chunked_transcribe(model, audio)
        result = {"segments": segments, "language": "es"}

        print(f"  ⏳ Alineando...")
        model_a, meta = whisperx.load_align_model(language_code="es", device=device)
        result = whisperx.align(
            result["segments"], model_a, meta, audio, device,
            return_char_alignments=False,
        )

        # --- Diarización: cadena de fallback ---
        diarize_df = None
        embeddings = {}
        emb_type = "whisperx_no_diarize"

        if BACKEND_SORTFORMER:
            print(f"  ⏳ Diarizando (Sortformer v2 + TitaNet)...")
            diarize_df, embeddings = _try_diarize_sortformer(path)
            if diarize_df is not None:
                emb_type = "titanet"

        if diarize_df is None and BACKEND_PYANNOTE:
            print(f"  ⏳ Diarizando (pyannote, speakers={min_speakers}-{max_speakers or 'auto'})...")
            try:
                from whisperx.diarize import DiarizationPipeline
                diarize_model = DiarizationPipeline(token=HF_TOKEN, device=device)
                # Pasar min/max speakers para que pyannote no fuerce un
                # número incorrecto de clusters. En batallas con host,
                # forzar 2 hace que el host se mezcle con un MC.
                diarize_kwargs = {
                    "min_speakers": min_speakers,
                    "return_embeddings": True,
                }
                if max_speakers is not None:
                    diarize_kwargs["max_speakers"] = max_speakers
                diarize_result = diarize_model(audio, **diarize_kwargs)

                if isinstance(diarize_result, tuple):
                    diarize_df, embeddings = diarize_result
                    embeddings = {
                        k: v.tolist() if hasattr(v, 'tolist') else list(v)
                        for k, v in embeddings.items()
                    }
                else:
                    diarize_df = diarize_result
                emb_type = "pyannote"
            except Exception as e:
                err_str = str(e)
                print(f"  ⚠️  pyannote falló: {err_str[:120]}")
                if "403" in err_str or "gated" in err_str:
                    print("  ⚠️  Token HF sin acceso. Acepta licencia en:")
                    print("      https://hf.co/pyannote/speaker-diarization-community-1")
                    print("      https://hf.co/pyannote/segmentation-3.0")

        if diarize_df is None:
            print(f"  ⚠️  Sin diarización — se aplicará fallback espectral")
            return result["segments"], {}, "whisperx_no_diarize"

        result = whisperx.assign_word_speakers(diarize_df, result)
        print(f"  ✅ WhisperX + diarización ({emb_type})")
        return result["segments"], embeddings, emb_type

    except Exception as e:
        err_str = str(e)
        print(f"  ⚠️  WhisperX falló: {err_str[:120]}")
        if "403" not in err_str and "gated" not in err_str:
            print("  🔄 Cambiando a fallback Whisper + Espectral...")
        return None, {}, "failed"


def transcribe_whisper_fallback(path):
    """Fallback: Whisper estándar + diarización espectral MFCC.

    Returns:
        (segments, embeddings, emb_type)
    """
    print(f"\n📝 [PASO 5-Fallback] Whisper + Espectral...")
    try:
        import whisper
        y, _ = librosa.load(path, sr=16000, mono=True)

        model = whisper.load_model(WHISPER_FALLBACK_MODEL)
        result = model.transcribe(y, language="es")

        segs = []
        for s in result["segments"]:
            s["speaker"] = "UNKNOWN"
            segs.append(s)

        segs, mfcc_embeddings = spectral_diarize(path, segs)
        return segs, mfcc_embeddings, "mfcc"
    except Exception as e:
        print(f"  ❌ Fallback también falló: {e}")
        import traceback
        traceback.print_exc()
        return [], {}, "failed"


def spectral_diarize(path, segments, num_speakers=2):
    """Diarización por clustering espectral (MFCC + KMeans).

    Extrae coeficientes MFCC promedio de cada segmento y agrupa
    con KMeans. Cada voz tiene un "color" tímbrico distinto reflejado
    en los coeficientes cepstrales.

    Returns:
        (segments_con_speaker, mfcc_embeddings)
    """
    from sklearn.cluster import KMeans

    print(f"  🔬 Diarización espectral (MFCC + KMeans, {num_speakers} speakers)...")
    y, sr = librosa.load(path, sr=SR, mono=True)

    features = []
    valid_indices = []
    for i, seg in enumerate(segments):
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        if end_sample - start_sample < sr * 0.3:
            continue
        chunk = y[start_sample:end_sample]
        if len(chunk) < 512:
            continue
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20)
        feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        features.append(feat)
        valid_indices.append(i)

    if len(features) < num_speakers:
        print(f"  ⚠️  Pocos segmentos ({len(features)}), no se puede clusterizar.")
        return segments, {}

    X = np.array(features)
    kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    label_map = {}
    for idx, seg_idx in enumerate(valid_indices):
        cluster = labels[idx]
        if cluster not in label_map:
            label_map[cluster] = f"SPEAKER_{len(label_map):02d}"
        segments[seg_idx]['speaker'] = label_map[cluster]

    for i, seg in enumerate(segments):
        if seg.get('speaker', 'UNKNOWN') == 'UNKNOWN' and valid_indices:
            closest = min(valid_indices,
                          key=lambda vi: abs(segments[vi]['start'] - seg['start']))
            seg['speaker'] = segments[closest]['speaker']

    counts = {}
    for s in segments:
        spk = s.get('speaker', 'UNKNOWN')
        counts[spk] = counts.get(spk, 0) + 1
    print(f"  ✅ Speakers: {counts}")

    mfcc_embeddings = {}
    for cluster_id, spk_label in label_map.items():
        mfcc_embeddings[spk_label] = kmeans.cluster_centers_[cluster_id].tolist()

    return segments, mfcc_embeddings
