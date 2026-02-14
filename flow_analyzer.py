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
import json
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
PENDINGS_DIR = os.path.join(SCRIPT_DIR, "pendings")
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

VOICEDB_PATH = os.path.join(SCRIPT_DIR, "voicedb.json")
# Umbral de similitud coseno para reconocer un MC conocido
# 0.75 para embeddings de pyannote (alta discriminación)
# 0.65 se usa internamente para embeddings MFCC (menos discriminativos)
EMBEDDING_MATCH_THRESHOLD = 0.75
MFCC_MATCH_THRESHOLD = 0.65

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
    """Busca archivo de audio a procesar.

    Prioridad:
    1. Cualquier archivo de audio en la carpeta pendings/
    2. Archivo 'batalla.*' en el directorio raíz (compatibilidad)
    """
    # 1. Buscar en pendings/ — toma el primero que encuentre
    if os.path.isdir(PENDINGS_DIR):
        for fname in os.listdir(PENDINGS_DIR):
            if any(fname.lower().endswith(ext) for ext in INPUT_EXTENSIONS):
                return os.path.join(PENDINGS_DIR, fname)

    # 2. Fallback: buscar batalla.* en raíz
    for ext in INPUT_EXTENSIONS:
        f = os.path.join(SCRIPT_DIR, INPUT_BASENAME + ext)
        if os.path.isfile(f): return f
    return None

def separate_audio(input_path):
    print("\n🧠 [PASO 0] Demucs (Separación)...")
    _patch_torchaudio()

    # Check cache — comparar fecha de modificación del archivo fuente
    # contra los stems. Si el fuente es más nuevo, re-procesar.
    fname = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(DEMUCS_OUTPUT_DIR, DEMUCS_MODEL, fname)
    voc = os.path.join(out_dir, "vocals.wav")
    inst = os.path.join(out_dir, "no_vocals.wav")

    if os.path.exists(voc) and os.path.exists(inst):
        source_mtime = os.path.getmtime(input_path)
        cache_mtime = min(os.path.getmtime(voc), os.path.getmtime(inst))
        if cache_mtime > source_mtime:
            print("  ✅ Archivos cacheados encontrados. Usando existentes.")
            return inst, voc
        else:
            print("  🔄 Archivo fuente más reciente que el cache. Re-procesando...")

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
# BASE DE DATOS DE VOCES (IDENTIFICACIÓN DE MCs)
# ============================================================================

def load_voicedb():
    """Carga la base de datos de huellas vocales desde voicedb.json.

    Retorna un dict {nombre_mc: {"embedding": [...], "battles": N}}
    o un dict vacío si el archivo no existe.
    """
    if os.path.exists(VOICEDB_PATH):
        try:
            with open(VOICEDB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  ⚠️  Error leyendo voicedb.json: {e}")
    return {}

def save_voicedb(db):
    """Persiste la base de datos de huellas vocales a voicedb.json."""
    with open(VOICEDB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def cosine_similarity(a, b):
    """Similitud coseno entre dos vectores.

    Mide el ángulo entre los vectores de embedding — cuanto más cercano a 1,
    más similares son las voces. Se usa en vez de distancia euclidiana porque
    los embeddings de speaker viven en un espacio donde la dirección importa
    más que la magnitud.
    """
    a, b = np.array(a), np.array(b)
    # No se pueden comparar embeddings de distinta dimensión
    # (ej. pyannote=256 dims vs MFCC=40 dims)
    if a.shape != b.shape:
        return 0.0
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def identify_speakers(embeddings, voicedb, is_mfcc=False):
    """Mapea SPEAKER_XX → nombre real comparando embeddings con la DB.

    Para cada speaker con embedding, calcula similitud coseno contra todos
    los MCs registrados. Si supera el umbral, asigna el nombre conocido.

    Args:
        embeddings: dict {speaker_label: [floats]} — embeddings del audio actual
        voicedb: dict cargado de voicedb.json
        is_mfcc: si True, usa umbral más bajo (MFCC menos discriminativo)

    Returns:
        speaker_map: dict {speaker_label: nombre_real_o_"SPEAKER_XX (Nuevo)"}
    """
    threshold = MFCC_MATCH_THRESHOLD if is_mfcc else EMBEDDING_MATCH_THRESHOLD
    speaker_map = {}

    # Evitar asignar el mismo MC a dos speakers distintos
    used_names = set()

    for spk_label, emb in embeddings.items():
        best_name = None
        best_sim = -1

        for mc_name, mc_data in voicedb.items():
            if mc_name in used_names:
                continue
            sim = cosine_similarity(emb, mc_data["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_name = mc_name

        if best_name and best_sim >= threshold:
            speaker_map[spk_label] = best_name
            used_names.add(best_name)
            print(f"  🎯 {spk_label} → {best_name} (similitud: {best_sim:.3f})")
        else:
            speaker_map[spk_label] = f"{spk_label} (Nuevo)"
            if best_name:
                print(f"  ❓ {spk_label} no coincide con nadie (mejor: {best_name} @ {best_sim:.3f})")
            else:
                print(f"  ❓ {spk_label} — no hay MCs registrados en la DB")

    return speaker_map

def register_new_speakers(embeddings, speaker_map, voicedb, is_mfcc=False):
    """Pregunta al usuario si quiere registrar speakers no identificados.

    Para cada speaker marcado como "(Nuevo)", ofrece al usuario la opción
    de darle un nombre. Si el MC ya existía en la DB, promedia el embedding
    viejo con el nuevo (media móvil) para robustecer el reconocimiento.
    Guarda el tipo de embedding ("pyannote" o "mfcc") para no mezclar
    dimensiones distintas en futuras comparaciones.

    Returns:
        voicedb actualizada (también se guarda a disco)
    """
    emb_type = "mfcc" if is_mfcc else "pyannote"
    nuevos = {label: name for label, name in speaker_map.items()
              if name.endswith("(Nuevo)")}

    if not nuevos:
        return voicedb

    print("\n" + "="*60)
    print("  📋 REGISTRO DE MCs")
    print("="*60)
    print("  Hay speakers no identificados. ¿Quieres registrarlos?")
    print("  (Deja vacío para saltar)\n")

    for spk_label in nuevos:
        if spk_label not in embeddings:
            continue
        nombre = input(f"  Nombre para {spk_label}: ").strip()
        if not nombre:
            continue

        emb = embeddings[spk_label]
        if nombre in voicedb and voicedb[nombre].get("type") == emb_type:
            # Media móvil: promediar embedding existente con el nuevo
            # Solo si son del mismo tipo (misma dimensión)
            old_emb = np.array(voicedb[nombre]["embedding"])
            new_emb = np.array(emb)
            n = voicedb[nombre]["battles"]
            # Media ponderada: el embedding existente tiene peso proporcional
            # al número de batallas previas
            averaged = ((old_emb * n) + new_emb) / (n + 1)
            voicedb[nombre]["embedding"] = averaged.tolist()
            voicedb[nombre]["battles"] = n + 1
            print(f"  ✅ {nombre} actualizado (batalla #{n+1})")
        else:
            # Nuevo MC, o el MC existía con otro tipo de embedding —
            # se reemplaza con el nuevo (pyannote > mfcc en calidad)
            if nombre in voicedb:
                print(f"  ⚠️  {nombre} tenía embedding tipo '{voicedb[nombre].get('type','?')}', reemplazando con '{emb_type}'")
            voicedb[nombre] = {
                "embedding": list(emb) if not isinstance(emb, list) else emb,
                "type": emb_type,
                "battles": 1
            }
            print(f"  ✅ {nombre} registrado en la DB ({emb_type}, {len(emb)} dims)")

        # Actualizar el speaker_map para que el scoreboard use el nombre real
        speaker_map[spk_label] = nombre

    save_voicedb(voicedb)
    return voicedb

def rename_segments(segments, speaker_map):
    """Reemplaza SPEAKER_XX por nombres reales en todos los segmentos."""
    for seg in segments:
        old_spk = seg.get("speaker", "UNKNOWN")
        if old_spk in speaker_map:
            seg["speaker"] = speaker_map[old_spk]
    return segments

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
        # return_embeddings=True para obtener las huellas vocales de cada speaker
        print(f"  ⏳ Diarizando (Pyannote)...")
        from whisperx.diarize import DiarizationPipeline
        diarize_model = DiarizationPipeline(token=HF_TOKEN, device=device)
        # min_speakers=2 porque una batalla de freestyle tiene al menos 2 MCs
        diarize_segs = diarize_model(audio, min_speakers=2, return_embeddings=True)

        # Cuando return_embeddings=True, diarize_model retorna una tupla
        # (diarize_df, {speaker_label: embedding_vector})
        embeddings = {}
        if isinstance(diarize_segs, tuple):
            diarize_segs, embeddings = diarize_segs
            # Convertir embeddings numpy a listas para serialización
            embeddings = {k: v.tolist() if hasattr(v, 'tolist') else list(v)
                          for k, v in embeddings.items()}

        result = whisperx.assign_word_speakers(diarize_segs, result)

        print("  ✅ WhisperX éxito.")
        return result["segments"], embeddings
        
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
        return None, {}

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
        return segments, {}

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

    # Generar embeddings simplificados: centroides MFCC por cluster
    # Cada centroide es el vector MFCC promedio de todos los segmentos de ese speaker,
    # útil como huella vocal básica cuando no hay pyannote disponible
    mfcc_embeddings = {}
    for cluster_id, spk_label in label_map.items():
        mfcc_embeddings[spk_label] = kmeans.cluster_centers_[cluster_id].tolist()

    return segments, mfcc_embeddings


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
        segs, mfcc_embeddings = _spectral_diarize(path, segs, num_speakers=2)

        return segs, mfcc_embeddings
    except Exception as e:
        print(f"  ❌ Fallback también falló: {e}")
        import traceback
        traceback.print_exc()
        return [], {}

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

    # 2. Cargar base de datos de voces conocidas
    voicedb = load_voicedb()
    if voicedb:
        print(f"\n📂 VoiceDB cargada: {len(voicedb)} MC(s) registrados — {', '.join(voicedb.keys())}")
    else:
        print(f"\n📂 VoiceDB vacía — los MCs se podrán registrar tras el análisis")

    # 3. Transcribe (Hybrid)
    segments, embeddings = try_transcribe_whisperx(vox)
    method = "WhisperX (Diarized)"
    is_mfcc = False
    if segments is None:
        segments, embeddings = transcribe_fallback_whisper(vox)
        method = "Whisper + Spectral Diarization (Fallback)"
        is_mfcc = True  # Los embeddings MFCC son menos discriminativos

    # 4. Identificar speakers por huella vocal
    speaker_map = {}
    if embeddings:
        print(f"\n🔍 Identificando speakers...")
        speaker_map = identify_speakers(embeddings, voicedb, is_mfcc=is_mfcc)
        segments = rename_segments(segments, speaker_map)

    save_txt(segments, os.path.basename(infile), method)

    # 5. Metrics per Speaker
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

    # 6. Scoreboard
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
        
        print(f"  {spk:<16} | SPS Max: {msp:<2.0f} | Acc: {acc:<4.1f}% | Syl: {tot}")
        if rhyme: print(f"    🔥 \"{rhyme[:50]}...\"")
        print("-" * 60)

    # 7. Registro de nuevos MCs (si hay embeddings disponibles)
    if embeddings:
        voicedb = register_new_speakers(embeddings, speaker_map, voicedb, is_mfcc=is_mfcc)
        # Re-renombrar segmentos por si el usuario registró nombres nuevos
        segments = rename_segments(segments, speaker_map)
        # Re-guardar transcripción con nombres actualizados
        save_txt(segments, os.path.basename(infile), method)

    # 8. Graph
    print(f"\n📊 Generando gráfico...")
    plt.figure(figsize=(14, 5), facecolor='#0d1117')
    ax = plt.gca()
    ax.set_facecolor('#161b22')
    ax.plot(np.linspace(0, len(y_vox)/sr, len(y_vox)), y_vox, color='#8b949e', alpha=0.3)
    for b in beats: ax.axvline(b, color='#ff6b6b', alpha=0.5, ls='--')
    # Paleta de colores para speakers con nombre real (que no están en SPEAKER_COLORS)
    _extra_colors = ["#3fb950", "#58a6ff", "#d2a8ff", "#f0883e", "#ff7b72"]
    _color_idx = 0
    for spk, onsets in speaker_onsets.items():
        if spk in SPEAKER_COLORS:
            c = SPEAKER_COLORS[spk]
        else:
            c = _extra_colors[_color_idx % len(_extra_colors)]
            _color_idx += 1
        ax.vlines(onsets, -0.8, 0.8, color=c, alpha=0.8, label=spk)
    ax.legend()
    plt.savefig(OUTPUT_GRAPH_PATH)
    print("  ✅ Done.")

if __name__ == "__main__":
    main()
