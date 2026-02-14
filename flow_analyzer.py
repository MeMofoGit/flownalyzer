"""
FlowMetrics v0.5 — Telemetría de Flow (Híbrido: WhisperX / Whisper)
=======================================
Sistema de telemetría acústica para batallas de freestyle rap.

CARACTERÍSTICAS V0.5:
1. Intenta usar WhisperX para Diarización (separar MCs).
2. Si falla (por incompatibilidad de Python/Windows), hace FALLBACK
   automático a Whisper estándar (sin separación de speakers).
3. Mantiene funciones de Flow (SPS, Beat Matching, Gráficas).

REQUISITOS:
- Recomendado: Python 3.10 (para WhisperX)
- Actual: Python 3.13 (funciona en modo Fallback)
"""

import os
import sys
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import gc
# NOTE: whisperx y torch se importan localmente para evitar crashes

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg"]
INPUT_BASENAME = "batalla"
DEMUCS_MODEL = "htdemucs"
DEMUCS_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "separated")
FALLBACK_INSTRUMENTAL = os.path.join(SCRIPT_DIR, "instrumental.wav")
FALLBACK_ACAPELLA = os.path.join(SCRIPT_DIR, "acapella.wav")
OUTPUT_GRAPH_PATH = os.path.join(SCRIPT_DIR, "flow_graph.png")
OUTPUT_TRANSCRIPT_PATH = os.path.join(SCRIPT_DIR, "transcripcion.txt")
SR = 22050
BEAT_MATCH_THRESHOLD_S = 0.05
WHISPERX_MODEL = "large-v2"
WHISPER_FALLBACK_MODEL = "small"
COMPUTE_TYPE = "int8"
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

SPEAKER_COLORS = {
    "SPEAKER_00": "#3fb950",  # Verde
    "SPEAKER_01": "#58a6ff",  # Azul
    "UNKNOWN":    "#8b949e"   # Gris
}

# ============================================================================
# PARCHES Y UTILIDADES
# ============================================================================

def _patch_torchaudio():
    """Parche crítico para Windows/Demucs."""
    try:
        import torchaudio as ta
        import torch
        import soundfile as sf
        
        if hasattr(ta, '_patched_by_flowmetrics'): return
        
        # Intentar cargar backend oficial
        try:
            from torchcodec.decoders import AudioDecoder
            return
        except: pass

        # Monkey-patch con soundfile
        def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                     channels_first=True, format=None, buffer_size=4096, backend=None):
            data, sr = sf.read(str(uri), dtype='float32', start=frame_offset,
                               stop=frame_offset+num_frames if num_frames > 0 else None,
                               always_2d=True)
            tensor = torch.from_numpy(data.T)
            if not channels_first: tensor = tensor.T
            return tensor, sr

        def _sf_save(uri, src, sample_rate, channels_first=True, **kwargs):
            import numpy as np
            if isinstance(src, torch.Tensor): data = src.detach().cpu().numpy()
            else: data = np.asarray(src)
            if channels_first and data.ndim == 2: data = data.T
            sf.write(str(uri), data, sample_rate)

        ta.load = _sf_load
        ta.save = _sf_save
        ta._patched_by_flowmetrics = True
    
    except Exception as e:
        # Fallback extremo: Mockear torchaudio si falla el import (común en Win/Py3.13)
        import types
        dummy = types.ModuleType("torchaudio")
        sys.modules["torchaudio"] = dummy
        dummy.load = lambda *a,**k: None
        dummy.save = lambda *a,**k: None
        try:
            import soundfile as sf
            import torch
            def _mock_load(uri, *a, **k):
                d, s = sf.read(str(uri), dtype='float32', always_2d=True)
                return torch.from_numpy(d.T), s
            dummy.load = _mock_load
        except: pass

def find_input_file():
    for ext in INPUT_EXTENSIONS:
        f = os.path.join(SCRIPT_DIR, INPUT_BASENAME + ext)
        if os.path.isfile(f): return f
    return None

def separate_audio(input_path):
    print("\n🧠 [PASO 0] Demucs (Separación)...")
    _patch_torchaudio()
    
    # Check cache
    fname = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(DEMUCS_OUTPUT_DIR, DEMUCS_MODEL, fname)
    voc = os.path.join(out_dir, "vocals.wav")
    inst = os.path.join(out_dir, "no_vocals.wav")
    
    if os.path.exists(voc) and os.path.exists(inst):
        print("  ✅ Archivos cacheados encontrados. Usando existentes.")
        return inst, voc

    print("  ⏳ Procesando separación (esto tarda)...")
    try:
        from demucs.separate import main as demucs_main
        sys.argv = ["demucs", "-n", DEMUCS_MODEL, "--two-stems=vocals", "-o", DEMUCS_OUTPUT_DIR, input_path]
        demucs_main()
        return inst, voc
    except Exception as e:
        print(f"  ❌ Error Demucs: {e}")
        sys.exit(1)

# ============================================================================
# ANÁLISIS DE AUDIO
# ============================================================================

def load_audio(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    return y, sr

def detect_beats(y, sr):
    print("\n🥁 [PASO 1] Análisis Instrumental...")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, 'item'): tempo = tempo.item()
    elif isinstance(tempo, np.ndarray): tempo = tempo[0]
    return float(tempo), librosa.frames_to_time(beats, sr=sr)

def generate_grid(tempo, beats, duration):
    print(f"\n🔲 [PASO 1b] Grid de Semicorcheas...")
    sub_dur = 60.0 / tempo / 4
    grid = []
    t = 0
    beat_idx = 0
    while t < duration:
        if beat_idx < len(beats) and abs(t - beats[beat_idx]) < sub_dur * 0.5:
            t = beats[beat_idx] # Resync
            beat_idx += 1
        grid.append(t)
        t += sub_dur
    return np.unique(np.array(grid))

def detect_onsets(y, sr):
    print("\n🎤 [PASO 2] Detección de Sílabas...")
    onsets = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, wait=1, delta=0.07)
    return librosa.frames_to_time(onsets, sr=sr)

def calc_metrics(onsets, grid, window=1.0):
    if len(onsets) == 0: return 0.0, 0.0, 0.0, 0, 0
    
    # SPS Max
    max_sps = 0
    peak_t = 0
    for t in onsets:
        c = np.searchsorted(onsets, t + window) - np.searchsorted(onsets, t)
        if c > max_sps: max_sps, peak_t = c, t
    
    # Beat Match
    hits = 0
    sorted_grid = np.sort(grid)
    for t in onsets:
        idx = np.searchsorted(sorted_grid, t)
        d = float('inf')
        if idx < len(sorted_grid): d = min(d, abs(sorted_grid[idx] - t))
        if idx > 0: d = min(d, abs(sorted_grid[idx-1] - t))
        if d <= BEAT_MATCH_THRESHOLD_S: hits += 1
    
    acc = (hits / len(onsets)) * 100
    return max_sps, peak_t, acc, hits, len(onsets)

# ============================================================================
# TRANSCRIPCIÓN HÍBRIDA (WHISPERX / WHISPER)
# ============================================================================

def try_transcribe_whisperx(path):
    print(f"\n📝 [PASO 5] Intentando WhisperX (Diarización)...")
    try:
        import torch
        import whisperx
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"  ⏳ Cargando modelo {WHISPERX_MODEL} en {device}...")
        model = whisperx.load_model(WHISPERX_MODEL, device, compute_type=COMPUTE_TYPE, language="es")
        audio = whisperx.load_audio(path)
        
        print(f"  ⏳ Transcribiendo (idioma: español)...")
        result = model.transcribe(audio, batch_size=16, language="es")
        
        # Align
        print(f"  ⏳ Alineando...")
        model_a, meta = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, meta, audio, device, return_char_alignments=False)
        
        # Diarize — En whisperx 3.8+ DiarizationPipeline se movió a whisperx.diarize
        # y el parámetro se renombró de use_auth_token a token
        print(f"  ⏳ Diarizando (Pyannote)...")
        from whisperx.diarize import DiarizationPipeline
        diarize_model = DiarizationPipeline(token=HF_TOKEN, device=device)
        # min_speakers=2 porque una batalla de freestyle tiene al menos 2 MCs
        diarize_segs = diarize_model(audio, min_speakers=2)
        result = whisperx.assign_word_speakers(diarize_segs, result)
        
        print("  ✅ WhisperX éxito.")
        return result["segments"]
        
    except Exception as e:
        err_str = str(e)
        print(f"  ⚠️  WhisperX falló: {err_str[:120]}")
        if "403" in err_str or "gated" in err_str:
            print("  ⚠️  Tu token HF no tiene acceso al modelo de diarización.")
            print("  ⚠️  Acepta la licencia en: https://hf.co/pyannote/speaker-diarization-community-1")
            print("  ⚠️  Y también en: https://hf.co/pyannote/segmentation-3.0")
        else:
            print("  ⚠️  (Probablemente incompatibilidad de torchaudio/Windows/Python 3.13)")
        print("  🔄 Cambiando a FALLBACK: Whisper + Diarización Espectral...")
        return None

def _spectral_diarize(path, segments, num_speakers=2):
    """Diarización por clustering espectral (MFCC + KMeans).

    Extrae los coeficientes MFCC promedio de cada segmento transcrito
    y agrupa los segmentos en N speakers usando KMeans. Esto funciona
    porque cada voz tiene un "color" tímbrico distinto que se refleja
    en los coeficientes cepstrales.
    """
    from sklearn.cluster import KMeans

    print(f"  🔬 Diarización espectral (MFCC + KMeans, {num_speakers} speakers)...")
    y, sr = librosa.load(path, sr=SR, mono=True)
    duration = len(y) / sr

    # Extraer MFCC promedio por cada segmento de la transcripción
    features = []
    valid_indices = []
    for i, seg in enumerate(segments):
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        # Evitar segmentos muy cortos que no dan MFCC estable
        if end_sample - start_sample < sr * 0.3:
            continue
        chunk = y[start_sample:end_sample]
        if len(chunk) < 512:
            continue
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=20)
        # Promedio + desviación estándar como feature vector (40 dims)
        feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        features.append(feat)
        valid_indices.append(i)

    if len(features) < num_speakers:
        print(f"  ⚠️  Pocos segmentos válidos ({len(features)}), no se puede clusterizar.")
        return segments

    X = np.array(features)
    kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Asignar labels a segmentos válidos
    label_map = {}
    for idx, seg_idx in enumerate(valid_indices):
        cluster = labels[idx]
        if cluster not in label_map:
            label_map[cluster] = f"SPEAKER_{len(label_map):02d}"
        segments[seg_idx]['speaker'] = label_map[cluster]

    # Segmentos no clusterizados: asignar por vecino más cercano temporalmente
    for i, seg in enumerate(segments):
        if seg.get('speaker', 'UNKNOWN') == 'UNKNOWN' and valid_indices:
            # Buscar el segmento válido más cercano en tiempo
            closest = min(valid_indices, key=lambda vi: abs(segments[vi]['start'] - seg['start']))
            seg['speaker'] = segments[closest]['speaker']

    # Contar segmentos por speaker
    counts = {}
    for s in segments:
        spk = s.get('speaker', 'UNKNOWN')
        counts[spk] = counts.get(spk, 0) + 1
    print(f"  ✅ Speakers detectados: {counts}")

    return segments


def transcribe_fallback_whisper(path):
    print(f"\n📝 [PASO 5-Fallback] Whisper + Diarización Espectral...")
    try:
        import whisper
        import librosa
        y, _ = librosa.load(path, sr=16000, mono=True)

        model = whisper.load_model(WHISPER_FALLBACK_MODEL)
        result = model.transcribe(y, language="es")

        # Inicializar como UNKNOWN, luego clusterizar
        segs = []
        for s in result["segments"]:
            s["speaker"] = "UNKNOWN"
            segs.append(s)

        # Aplicar diarización espectral (MFCC + KMeans)
        segs = _spectral_diarize(path, segs, num_speakers=2)

        return segs
    except Exception as e:
        print(f"  ❌ Fallback también falló: {e}")
        import traceback
        traceback.print_exc()
        return []

def save_txt(segments, input_name, method):
    with open(OUTPUT_TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(f"FlowMetrics v0.5 — {method}\n")
        f.write(f"Source: {input_name}\n\n")
        for s in segments:
            f.write(f"[{s.get('speaker','?')}] [{int(s['start']//60):02d}:{s['start']%60:04.1f}] {s['text'].strip()}\n")
    print(f"  📄 Saved: {OUTPUT_TRANSCRIPT_PATH}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    _patch_torchaudio() # Patch init
    
    print("="*60 + "\n  🎤 FlowMetrics v0.5 (Hybrid Engine)\n" + "="*60)
    
    infile = find_input_file()
    if not infile:
        if os.path.exists(FALLBACK_INSTRUMENTAL): infile, inst, vox = "Pre-separated", FALLBACK_INSTRUMENTAL, FALLBACK_ACAPELLA
        else: sys.exit("❌ Faltan archivos de audio")
    else:
        inst, vox = separate_audio(infile)

    # 1. Analyze
    y_inst, sr = load_audio(inst)
    tempo, beats = detect_beats(y_inst, sr)
    grid = generate_grid(tempo, beats, librosa.get_duration(y=y_inst, sr=sr))
    
    y_vox, sr_vox = load_audio(vox)
    all_onsets = detect_onsets(y_vox, sr_vox)

    # 2. Transcribe (Hybrid)
    segments = try_transcribe_whisperx(vox)
    method = "WhisperX (Diarized)"
    if segments is None:
        segments = transcribe_fallback_whisper(vox)
        method = "Whisper + Spectral Diarization (Fallback)"

    save_txt(segments, os.path.basename(infile), method)

    # 3. Metrics per Speaker
    speaker_onsets = {}
    
    # Mapear onsets a speakers usando los rangos temporales de la transcripción
    # Funciona tanto con WhisperX como con el fallback espectral
    has_speakers = any(s.get("speaker", "UNKNOWN") != "UNKNOWN" for s in segments)
    if has_speakers:
        ranges = {}
        for s in segments:
            spk = s.get("speaker", "UNKNOWN")
            if spk not in ranges: ranges[spk] = []
            ranges[spk].append((s['start'], s['end']))

        for onset in all_onsets:
            assigned = False
            for spk, rngs in ranges.items():
                for (st, en) in rngs:
                    if st-0.1 <= onset <= en+0.1:
                        if spk not in speaker_onsets: speaker_onsets[spk] = []
                        speaker_onsets[spk].append(onset)
                        assigned = True
                        break
                if assigned: break
            if not assigned:
                if "UNKNOWN" not in speaker_onsets: speaker_onsets["UNKNOWN"] = []
                speaker_onsets["UNKNOWN"].append(onset)
    else:
        speaker_onsets["UNKNOWN"] = list(all_onsets)

    # 4. Scoreboard
    print("\n" + "="*60 + "\n  🏆 SCOREBOARD\n" + "="*60)
    for spk, onsets in speaker_onsets.items():
        if not onsets: continue
        msp, pt, acc, h, tot = calc_metrics(np.array(onsets), grid)
        
        # Find rhyme
        rhyme = ""
        for s in segments:
            if s.get("speaker") == spk and s['start'] <= pt <= s['end']+0.5:
                rhyme = s['text'].strip()
                break
        
        print(f"  {spk:<12} | SPS Max: {msp:<2.0f} | Acc: {acc:<4.1f}% | Syl: {tot}")
        if rhyme: print(f"    🔥 \"{rhyme[:50]}...\"")
        print("-" * 60)

    # 5. Graph
    print(f"\n📊 Generando gráfico...")
    plt.figure(figsize=(14, 5), facecolor='#0d1117')
    ax = plt.gca()
    ax.set_facecolor('#161b22')
    ax.plot(np.linspace(0, len(y_vox)/sr, len(y_vox)), y_vox, color='#8b949e', alpha=0.3)
    for b in beats: ax.axvline(b, color='#ff6b6b', alpha=0.5, ls='--')
    for spk, onsets in speaker_onsets.items():
        c = SPEAKER_COLORS.get(spk, SPEAKER_COLORS["UNKNOWN"])
        ax.vlines(onsets, -0.8, 0.8, color=c, alpha=0.8, label=spk)
    ax.legend()
    plt.savefig(OUTPUT_GRAPH_PATH)
    print("  ✅ Done.")

if __name__ == "__main__":
    main()
