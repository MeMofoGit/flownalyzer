"""
FlowMetrics v0.7 — Telemetría de Flow + Biometría DSP
=======================================
Sistema de telemetría acústica para batallas de freestyle rap.

CARACTERÍSTICAS V0.7:
1. Diarización con cadena de fallback:
   a) NVIDIA Sortformer v2 + TitaNet (DER ~6.57%, requiere NeMo en WSL2)
   b) pyannote vía WhisperX (DER ~19.9%)
   c) Diarización espectral MFCC + KMeans (fallback sin GPU/modelos)
2. Transcripción siempre vía WhisperX (o Whisper estándar como fallback).
3. Identificación de MCs por huella vocal (TitaNet 192d / pyannote 256d / MFCC 40d).
4. Biometría de Flow (Fase 4):
   a) Micro-timing / Índice de Swing (laid-back vs métrico vs arrítmico)
   b) Contorno de Pitch / Índice Melódico (melódico vs agresivo)
   c) Envolvente RMS / Detección de Chicleo (sustain intencional)
   d) Flow Index combinado (0-100) con clasificación de estilo.

REQUISITOS:
- Óptimo: WSL2 + Python 3.10-3.12 + NeMo (para Sortformer v2)
- Recomendado: Python 3.10 (para WhisperX + pyannote)
- Mínimo: Python 3.13 (funciona en modo Fallback espectral)
"""

import os
import sys
import json
import warnings
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import gc

# Suprimir warnings ruidosos de torchcodec/pyannote (no afectan funcionalidad)
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*libtorchcodec.*")

# Suprimir warnings ruidosos de NeMo/Megatron (no afectan funcionalidad)
warnings.filterwarnings("ignore", message=".*Megatron.*")
warnings.filterwarnings("ignore", message=".*num_microbatches_calculator.*")
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")
warnings.filterwarnings("ignore", message=".*setup_training_data.*")
warnings.filterwarnings("ignore", message=".*setup_validation_data.*")
warnings.filterwarnings("ignore", message=".*setup_test_data.*")
warnings.filterwarnings("ignore", message=".*OneLogger.*")
warnings.filterwarnings("ignore", message=".*No exporters were provided.*")
warnings.filterwarnings("ignore", message=".*Xet Storage.*")
warnings.filterwarnings("ignore", message=".*upgraded your loaded checkpoint.*")

# Suprimir logs [NeMo W/I ...] y mensajes verbose de PyTorch Lightning
import logging as _logging
for _logger_name in ['nemo_logger', 'nemo', 'pytorch_lightning', 'lightning.fabric', 'lightning.pytorch']:
    _logging.getLogger(_logger_name).setLevel(_logging.ERROR)

# Windows: forzar UTF-8 para emojis en consola
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

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

# --- Modelos NVIDIA (requieren NeMo, funciona en WSL2) ---
SORTFORMER_MODEL_NAME = "nvidia/diar_streaming_sortformer_4spk-v2"
TITANET_MODEL_NAME = "nvidia/speakerverification_en_titanet_large"

# Umbrales de similitud coseno por tipo de embedding.
# Cada modelo genera embeddings en un espacio distinto con distinta
# capacidad discriminativa, por eso necesitan umbrales distintos.
TITANET_MATCH_THRESHOLD = 0.7    # TitaNet 192 dims — muy discriminativo
EMBEDDING_MATCH_THRESHOLD = 0.75  # pyannote 256 dims — alta discriminación
MFCC_MATCH_THRESHOLD = 0.65       # MFCC 40 dims — menos discriminativo

SPEAKER_COLORS = {
    "SPEAKER_00": "#3fb950",  # Verde
    "SPEAKER_01": "#58a6ff",  # Azul
    "UNKNOWN":    "#8b949e"   # Gris
}

# ============================================================================
# DETECCIÓN DE BACKENDS DISPONIBLES
# ============================================================================
# Se evalúan una sola vez al importar el módulo para saber qué pipeline
# de diarización está disponible. Esto evita try/except repetidos.

BACKEND_SORTFORMER = False
BACKEND_PYANNOTE = False

try:
    from nemo.collections.asr.models import SortformerEncLabelModel, EncDecSpeakerLabelModel
    BACKEND_SORTFORMER = True
except ImportError:
    pass

try:
    from whisperx.diarize import DiarizationPipeline as _DiarPipeline
    BACKEND_PYANNOTE = True
except ImportError:
    pass

print("[Backend] NeMo Sortformer v2 + TitaNet:", "DISPONIBLE ✅" if BACKEND_SORTFORMER else "no disponible")
print("[Backend] pyannote (vía WhisperX):", "DISPONIBLE ✅" if BACKEND_PYANNOTE else "no disponible")
if not BACKEND_SORTFORMER and not BACKEND_PYANNOTE:
    print("[Backend] Se usará fallback espectral (MFCC + KMeans)")

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
# BIOMETRÍA DE FLOW (Fase 4 — Análisis DSP avanzado)
# ============================================================================
# Estas métricas van más allá de "cuántas sílabas por segundo" y miden
# el *estilo* del flow: cómo se posiciona rítmicamente (swing), cómo
# entona (melodía vs agresión), y cómo sostiene las palabras (chicleo).

def extract_speaker_audio(y_vox, sr, segments, speaker_label):
    """Extrae el array numpy concatenado de un speaker desde vocals.wav.

    Recorta los fragmentos de audio correspondientes a los timestamps
    de WhisperX para el speaker indicado y los concatena en orden.
    Esto es necesario porque librosa.yin y librosa.feature.rms operan
    sobre arrays numpy, no sobre timestamps.

    Args:
        y_vox: señal de audio completa (numpy array mono)
        sr: sample rate
        segments: lista de segmentos con 'start', 'end', 'speaker'
        speaker_label: nombre del speaker a extraer

    Returns:
        numpy array con el audio concatenado del speaker,
        o array vacío si no hay segmentos
    """
    chunks = []
    for seg in segments:
        if seg.get("speaker") != speaker_label:
            continue
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)
        chunk = y_vox[start_sample:end_sample]
        if len(chunk) > 0:
            chunks.append(chunk)
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)


def _analizar_microtiming(onsets, grid):
    """Métrica 1: Análisis de Micro-timing (Índice de Swing).

    Para cada sílaba (onset), calcula la diferencia en milisegundos
    respecto a la semicorchea más cercana en la cuadrícula rítmica.

    La distribución de esos deltas revela el estilo rítmico:
    - σ bajo + μ entre 15-45ms → "Laid-back" (retraso intencional, swing)
    - σ bajo + μ < 15ms → Mecánicamente preciso (métrico)
    - σ alto (> 50ms) → Arrítmico (pierde el tempo)

    Returns:
        dict con: delta_media_ms, delta_std_ms, bonus_swing, swing_label
    """
    if len(onsets) == 0 or len(grid) == 0:
        return {
            "delta_media_ms": 0.0, "delta_std_ms": 0.0,
            "bonus_swing": 0.0, "swing_label": "Sin datos"
        }

    sorted_grid = np.sort(grid)
    deltas_ms = []

    for t in onsets:
        # Buscar la semicorchea más cercana usando búsqueda binaria
        idx = np.searchsorted(sorted_grid, t)
        d = float("inf")
        if idx < len(sorted_grid):
            d = min(d, abs(sorted_grid[idx] - t))
        if idx > 0:
            d = min(d, abs(sorted_grid[idx - 1] - t))
        # Convertir a milisegundos — trabajar en ms es más intuitivo
        # para músicos (un "swing" de 20ms es audible pero sutil)
        deltas_ms.append(d * 1000.0)

    deltas = np.array(deltas_ms)
    mu = float(np.mean(deltas))
    sigma = float(np.std(deltas))

    # Lógica de puntuación de swing
    bonus_swing = 0.0
    swing_label = "Neutro"

    if sigma < 25.0:
        # Consistente rítmicamente — verificar si hay swing intencional
        if 15.0 <= mu <= 45.0:
            # "Laid-back": el MC retrasa sistemáticamente sus sílabas
            # respecto al grid. Es un recurso estilístico muy valorado
            # en hip-hop (ej. Snoop Dogg, Biggie). Bonus proporcional
            # a qué tan centrado está en el "sweet spot" de 30ms.
            distance_to_sweet = abs(mu - 30.0)
            bonus_swing = max(0, 8.0 - (distance_to_sweet / 15.0) * 8.0)
            swing_label = "Laid-back"
        elif mu < 15.0:
            # Mecánicamente preciso — bueno pero sin groove
            bonus_swing = 3.0
            swing_label = "Metrico"
    elif sigma > 50.0:
        # Alta variabilidad → arritmia, penalizar
        bonus_swing = -5.0
        swing_label = "Arritmico"
    else:
        # Zona intermedia (25-50ms σ) — algo de variación, ni bonus ni penal
        bonus_swing = 0.0
        swing_label = "Variable"

    return {
        "delta_media_ms": round(mu, 1),
        "delta_std_ms": round(sigma, 1),
        "bonus_swing": round(bonus_swing, 1),
        "swing_label": swing_label,
    }


def _analizar_sincopa(onsets, grid):
    """Métrica 1b: Análisis de Síncopa (Contratiempos).

    En una cuadrícula de semicorcheas, las posiciones fuertes son la 1a y
    la 3a (tiempos fuertes del compás), y las débiles son la 2a y la 4a
    (contratiempos). Un MC que coloca muchas sílabas en contratiempos
    está sincopando — creando tensión rítmica y groove.

    Para cada onset, determinamos en qué semicorchea del grupo de 4 cae:
    - Posición 0 (beat fuerte) y 2 (subdivisión fuerte) → "fuertes"
    - Posición 1 y 3 (contratiempos) → "débiles" / sincopadas

    Un alto % de sílabas en posiciones débiles = bonus de síncopa.
    Esto premia a MCs como Wos que rapean "entre" los beats.

    Args:
        onsets: array de tiempos de onset del speaker
        grid: array de la cuadrícula de semicorcheas

    Returns:
        dict con: pct_sincopa (%), bonus_sincopa, sincopa_label
    """
    if len(onsets) == 0 or len(grid) < 4:
        return {
            "pct_sincopa": 0.0, "bonus_sincopa": 0.0,
            "sincopa_label": "Sin datos"
        }

    sorted_grid = np.sort(grid)
    en_debil = 0
    total_asignados = 0

    for t in onsets:
        # Encontrar la semicorchea más cercana
        idx = np.searchsorted(sorted_grid, t)
        best_idx = idx
        best_d = float("inf")
        if idx < len(sorted_grid):
            d = abs(sorted_grid[idx] - t)
            if d < best_d:
                best_d = d
                best_idx = idx
        if idx > 0:
            d = abs(sorted_grid[idx - 1] - t)
            if d < best_d:
                best_d = d
                best_idx = idx - 1

        # Solo contar si el onset está razonablemente cerca del grid
        # (dentro de ~60ms, que es ~1 semicorchea a 120bpm)
        if best_d > 0.060:
            continue

        total_asignados += 1
        # Posición dentro del grupo de 4 semicorcheas:
        # 0=beat fuerte, 1=contratiempo, 2=subdivisión fuerte, 3=contratiempo
        posicion = best_idx % 4
        if posicion in (1, 3):
            en_debil += 1

    if total_asignados == 0:
        return {
            "pct_sincopa": 0.0, "bonus_sincopa": 0.0,
            "sincopa_label": "Sin datos"
        }

    pct_sincopa = (en_debil / total_asignados) * 100.0

    # Bonus de síncopa:
    # En un flow perfectamente "en el beat", ~50% caería en fuertes y 50% en débiles
    # (distribuido uniforme). Pero la mayoría de MCs caen más en fuertes (~60-70%).
    # Un MC con >55% en débiles está sincopando activamente.
    # Escala: 0 pts si ≤40%, hasta 6 pts si ≥65%.
    if pct_sincopa >= 65.0:
        bonus_sincopa = 6.0
        sincopa_label = "Muy sincopado"
    elif pct_sincopa >= 40.0:
        bonus_sincopa = (pct_sincopa - 40.0) / 25.0 * 6.0
        sincopa_label = "Sincopado" if pct_sincopa >= 55.0 else "Ligero"
    else:
        bonus_sincopa = 0.0
        sincopa_label = "En el beat"

    return {
        "pct_sincopa": round(pct_sincopa, 1),
        "bonus_sincopa": round(bonus_sincopa, 1),
        "sincopa_label": sincopa_label,
    }


def _analizar_smooth_pitch(speaker_audio, sr):
    """Suavidad Melódica (Smooth Pitch) — mide intención musical real.

    El problema del rango vocal (P95-P5) es que un MC que GRITA en los
    punchlines genera picos extremos de F0 que inflan artificialmente
    el rango — pareciendo "melódico" cuando en realidad solo grita.

    La solución: medir la DERIVADA de F0 (diferencia frame a frame).
    - Un MC que entona (Wos) tiene transiciones suaves entre notas →
      derivada baja y continua, como una curva melódica.
    - Un MC que grita (Chuty) tiene saltos bruscos de frecuencia →
      derivada alta con picos.

    Métricas calculadas:
    - smooth_score: inverso de la mediana de |ΔF0|, normalizado.
      Mediana en vez de media para ser robusto a saltos aislados.
    - pct_voiced: % de frames con pitch válido (presencia tonal).
      Un MC que mantiene tono constantemente es más musical que uno
      que alterna entre hablar y gritar.

    Args:
        speaker_audio: numpy array mono del audio del speaker
        sr: sample rate

    Returns:
        dict con: smooth_score (0-100), pct_voiced, pitch_label, rango_vocal
    """
    resultado_vacio = {
        "smooth_score": 0.0, "pct_voiced": 0.0,
        "pitch_label": "Sin datos", "rango_vocal": 0.0,
        "median_delta_f0": 0.0,
    }

    if len(speaker_audio) < sr * 0.5:
        return resultado_vacio

    f0 = librosa.yin(
        speaker_audio, fmin=65, fmax=300, sr=sr,
        frame_length=2048, hop_length=512
    )

    # Filtrar frames sin voz (YIN retorna fmin o fmax en silencio)
    voiced_mask = (f0 > 66) & (f0 < 299)
    valid_f0 = f0[voiced_mask]
    pct_voiced = (np.sum(voiced_mask) / len(f0)) * 100.0 if len(f0) > 0 else 0.0

    if len(valid_f0) < 20:
        return resultado_vacio

    # Rango vocal (solo informativo, ya no se usa para scoring)
    p95 = float(np.percentile(valid_f0, 95))
    p5 = float(np.percentile(valid_f0, 5))
    rango_vocal = p95 - p5

    # Derivada de F0: diferencia entre frames adyacentes VOICED consecutivos.
    # Solo entre frames que ambos tienen pitch válido, para no contaminar
    # con saltos silencio→voz que no son transiciones melódicas reales.
    voiced_indices = np.where(voiced_mask)[0]
    consecutive_pairs = np.where(np.diff(voiced_indices) == 1)[0]

    if len(consecutive_pairs) < 10:
        return {
            "smooth_score": 0.0, "pct_voiced": round(pct_voiced, 1),
            "pitch_label": "Sin datos", "rango_vocal": round(rango_vocal, 1),
            "median_delta_f0": 0.0,
        }

    # |ΔF0| entre frames voiced consecutivos
    deltas_f0 = np.abs(np.diff(f0[voiced_indices[consecutive_pairs[0]:consecutive_pairs[-1]+2]]))

    # Mediana de |ΔF0| — robusto a outliers (un grito aislado no cambia la mediana)
    median_delta = float(np.median(deltas_f0))

    # Smooth score: mapear mediana de |ΔF0| a 0-100.
    # Valores observados empíricamente:
    #   - MC muy suave/melódico (Wos): mediana ~1-3 Hz (transiciones graduales)
    #   - MC expresivo normal:          mediana ~3-6 Hz
    #   - MC que grita/percusivo:       mediana ~6-15 Hz (saltos bruscos)
    # Escala invertida: menor delta = mayor smooth_score
    if median_delta <= 1.5:
        smooth_score = 100.0
    elif median_delta >= 12.0:
        smooth_score = 0.0
    else:
        # Interpolación lineal invertida: 1.5→100, 12→0
        smooth_score = (1.0 - (median_delta - 1.5) / 10.5) * 100.0

    # Clasificación
    if smooth_score >= 65.0 and pct_voiced >= 40.0:
        pitch_label = "Melodico"
    elif smooth_score >= 40.0:
        pitch_label = "Expresivo"
    else:
        pitch_label = "Agresivo"

    return {
        "smooth_score": round(smooth_score, 1),
        "pct_voiced": round(pct_voiced, 1),
        "pitch_label": pitch_label,
        "rango_vocal": round(rango_vocal, 1),
        "median_delta_f0": round(median_delta, 2),
    }


def _analizar_sustain_pct(speaker_audio, sr):
    """Sustain Real — % de tiempo en chicleo respecto al tiempo total.

    En vez de contar chicleos absolutos (que sesgaba a MCs ruidosos),
    mide el PORCENTAJE del tiempo total del MC que pasa en sustain.
    Un MC que chiclea el 20% de su turno tiene más groove que uno que
    chiclea el 5%, independientemente de su volumen.

    Umbral al 30% del pico RMS para captar susurros sostenidos.

    Args:
        speaker_audio: numpy array mono del audio del speaker
        sr: sample rate

    Returns:
        dict con: pct_sustain (%), chicleadas (int), varianza_rms, dinamica_label
    """
    resultado_vacio = {
        "pct_sustain": 0.0, "chicleadas": 0,
        "varianza_rms": 0.0, "dinamica_label": "Sin datos"
    }

    if len(speaker_audio) < sr * 0.5:
        return resultado_vacio

    rms = librosa.feature.rms(y=speaker_audio, frame_length=2048, hop_length=512)[0]

    if len(rms) == 0 or np.max(rms) == 0:
        return resultado_vacio

    umbral = np.max(rms) * 0.30
    hop_dur_s = 512.0 / sr
    min_frames = int(0.250 / hop_dur_s)

    above = rms > umbral
    chicleadas = 0
    racha = 0
    frames_en_sustain = 0

    for val in above:
        if val:
            racha += 1
        else:
            if racha >= min_frames:
                chicleadas += 1
                frames_en_sustain += racha
            racha = 0
    if racha >= min_frames:
        chicleadas += 1
        frames_en_sustain += racha

    # % de tiempo en sustain respecto al total de frames
    pct_sustain = (frames_en_sustain / len(rms)) * 100.0 if len(rms) > 0 else 0.0

    # Dinámica de volumen
    rms_norm = rms / np.max(rms)
    varianza_rms = float(np.var(rms_norm))

    if varianza_rms >= 0.10:
        dinamica_label = "Muy dinamico"
    elif varianza_rms >= 0.06:
        dinamica_label = "Dinamico"
    elif varianza_rms >= 0.03:
        dinamica_label = "Moderado"
    else:
        dinamica_label = "Constante"

    return {
        "pct_sustain": round(pct_sustain, 1),
        "chicleadas": chicleadas,
        "varianza_rms": round(varianza_rms, 4),
        "dinamica_label": dinamica_label,
    }


def analizar_biometria_flow(speaker_audio, onsets, grid, sr, sps_max=0, total_silabas=0):
    """Función principal de biometría — DOS índices independientes (0-100).

    Separar en dos ejes elimina el sesgo donde un MC robótico/gritón
    podía superar a uno musical porque las métricas se mezclaban.

    ═══════════════════════════════════════════════════════════
    ÍNDICE DE TÉCNICA (0-100) — "¿Qué tan preciso y rápido es?"
    ═══════════════════════════════════════════════════════════
    - Precisión rítmica: % de onsets que caen en el grid (peso 70%)
    - Velocidad: SPS máximo normalizado (peso 20%)
    - Volumen silábico: total de sílabas normalizado (peso 10%)

    Esto favorece a MCs rápidos, constantes y clavados en la cuadrícula.

    ═══════════════════════════════════════════════════════════
    ÍNDICE DE GROOVE (0-100) — "¿Qué tan musical y expresivo es?"
    ═══════════════════════════════════════════════════════════
    - Swing consistente: laid-back intencional (peso 25%)
    - Síncopa: % de sílabas en contratiempos (peso 25%)
    - Smooth Pitch: suavidad de transiciones melódicas (peso 25%)
    - Sustain Real: % del tiempo en chicleo (peso 15%)
    - Dinámica: varianza de volumen — no robótico (peso 10%)

    Esto favorece a MCs con groove, entonación suave y expresividad.

    Args:
        speaker_audio: numpy array mono del speaker
        onsets: array de tiempos de onset del speaker
        grid: array de la cuadrícula de semicorcheas
        sr: sample rate
        sps_max: sílabas por segundo máximo (de calc_metrics)
        total_silabas: total de sílabas detectadas

    Returns:
        dict con todas las métricas + indice_tecnica + indice_groove
    """
    # --- Sub-análisis ---
    timing = _analizar_microtiming(onsets, grid)
    sincopa = _analizar_sincopa(onsets, grid)
    pitch = _analizar_smooth_pitch(speaker_audio, sr)
    sustain = _analizar_sustain_pct(speaker_audio, sr)

    # ═══════════════════════════════════════════════════
    # ÍNDICE DE TÉCNICA (0-100)
    # ═══════════════════════════════════════════════════

    # A) Precisión rítmica (0-70 pts)
    if len(onsets) > 0 and len(grid) > 0:
        sorted_grid = np.sort(grid)
        hits = 0
        for t in onsets:
            idx = np.searchsorted(sorted_grid, t)
            d = float("inf")
            if idx < len(sorted_grid):
                d = min(d, abs(sorted_grid[idx] - t))
            if idx > 0:
                d = min(d, abs(sorted_grid[idx - 1] - t))
            if d <= BEAT_MATCH_THRESHOLD_S:
                hits += 1
        accuracy_pct = (hits / len(onsets)) * 100.0
        pts_precision = (accuracy_pct / 100.0) * 70.0
    else:
        accuracy_pct = 0.0
        pts_precision = 0.0

    # B) Velocidad — SPS máximo normalizado (0-20 pts)
    # 15 SPS es el techo práctico en freestyle (Chuty-level)
    # Por debajo de 5 SPS no hay bonus
    pts_velocidad = max(0.0, min(20.0, (sps_max - 5.0) / 10.0 * 20.0))

    # C) Volumen silábico — total de sílabas normalizado (0-10 pts)
    # 400 sílabas en una batalla es bastante; 100 es poco
    pts_volumen = max(0.0, min(10.0, (total_silabas - 100) / 300.0 * 10.0))

    indice_tecnica = max(0.0, min(100.0, pts_precision + pts_velocidad + pts_volumen))

    # ═══════════════════════════════════════════════════
    # ÍNDICE DE GROOVE (0-100)
    # ═══════════════════════════════════════════════════

    # A) Swing consistente (0-25 pts)
    # Laid-back con σ bajo = máximo groove rítmico
    # μ entre 20-50ms Y σ < 20ms → zona dorada de swing
    mu = timing["delta_media_ms"]
    sigma = timing["delta_std_ms"]
    if sigma < 20.0 and 20.0 <= mu <= 50.0:
        # Sweet spot: μ=35ms, σ lo más bajo posible
        swing_quality = 1.0 - (abs(mu - 35.0) / 25.0)
        consistency_bonus = 1.0 - (sigma / 20.0)
        pts_swing = max(0.0, (swing_quality * 0.6 + consistency_bonus * 0.4) * 25.0)
    elif sigma < 25.0 and mu < 20.0:
        # Métrico: preciso pero sin groove — algo de puntos
        pts_swing = 8.0
    elif sigma > 50.0:
        # Arrítmico: penalizar
        pts_swing = 0.0
    else:
        # Zona intermedia
        pts_swing = max(0.0, 12.0 - sigma * 0.2)

    # B) Síncopa (0-25 pts)
    # Mapear pct_sincopa (0-100%) a 0-25 pts
    # El % natural sin intención es ~35-45%. Premiar por encima de 45%.
    pct_s = sincopa["pct_sincopa"]
    if pct_s >= 65.0:
        pts_sincopa = 25.0
    elif pct_s >= 45.0:
        pts_sincopa = (pct_s - 45.0) / 20.0 * 25.0
    else:
        pts_sincopa = 0.0

    # C) Smooth Pitch (0-25 pts)
    # smooth_score ya está en 0-100, escalamos a 0-25
    # PERO: solo si hay suficiente presencia tonal (pct_voiced > 30%).
    # Si el MC casi no tiene frames voiced, su "suavidad" no es
    # musicalidad sino simplemente que no canta.
    if pitch["pct_voiced"] >= 30.0:
        pts_smooth = (pitch["smooth_score"] / 100.0) * 25.0
    else:
        pts_smooth = (pitch["smooth_score"] / 100.0) * 25.0 * (pitch["pct_voiced"] / 30.0)

    # D) Sustain Real (0-15 pts) — sin cap fijo
    # Mapear pct_sustain (0-50%) a 0-15 pts
    # Un 30% del tiempo en chicleo es mucho groove
    pts_sustain = max(0.0, min(15.0, (sustain["pct_sustain"] / 30.0) * 15.0))

    # E) Dinámica de volumen (0-10 pts)
    # Premiar variación, penalizar monotonía
    var = sustain["varianza_rms"]
    if var >= 0.10:
        pts_dinamica = 10.0
    elif var >= 0.03:
        pts_dinamica = (var - 0.03) / 0.07 * 10.0
    else:
        pts_dinamica = 0.0

    indice_groove = max(0.0, min(100.0,
        pts_swing + pts_sincopa + pts_smooth + pts_sustain + pts_dinamica
    ))

    return {
        # Micro-timing
        "delta_media_ms": timing["delta_media_ms"],
        "delta_std_ms": timing["delta_std_ms"],
        "swing_label": timing["swing_label"],
        # Síncopa
        "pct_sincopa": sincopa["pct_sincopa"],
        "sincopa_label": sincopa["sincopa_label"],
        # Pitch (Smooth)
        "smooth_score": pitch["smooth_score"],
        "pct_voiced": pitch["pct_voiced"],
        "rango_vocal": pitch["rango_vocal"],
        "median_delta_f0": pitch["median_delta_f0"],
        "pitch_label": pitch["pitch_label"],
        # Sustain + Dinámica
        "pct_sustain": sustain["pct_sustain"],
        "chicleadas": sustain["chicleadas"],
        "varianza_rms": sustain["varianza_rms"],
        "dinamica_label": sustain["dinamica_label"],
        # Desglose Técnica
        "accuracy_pct": round(accuracy_pct, 1),
        "pts_precision": round(pts_precision, 1),
        "pts_velocidad": round(pts_velocidad, 1),
        "pts_volumen": round(pts_volumen, 1),
        # Desglose Groove
        "pts_swing": round(pts_swing, 1),
        "pts_sincopa": round(pts_sincopa, 1),
        "pts_smooth": round(pts_smooth, 1),
        "pts_sustain": round(pts_sustain, 1),
        "pts_dinamica": round(pts_dinamica, 1),
        # Índices finales
        "indice_tecnica": round(indice_tecnica, 1),
        "indice_groove": round(indice_groove, 1),
    }


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

def identify_speakers(embeddings, voicedb, emb_type="pyannote"):
    """Mapea SPEAKER_XX → nombre real comparando embeddings con la DB.

    Para cada speaker con embedding, calcula similitud coseno contra todos
    los MCs registrados. Si supera el umbral, asigna el nombre conocido.

    Args:
        embeddings: dict {speaker_label: [floats]} — embeddings del audio actual
        voicedb: dict cargado de voicedb.json
        emb_type: tipo de embedding ("titanet", "pyannote", "mfcc")
            Determina el umbral de similitud a usar.

    Returns:
        speaker_map: dict {speaker_label: nombre_real_o_"SPEAKER_XX (Nuevo)"}
    """
    # Cada tipo de embedding tiene distinta capacidad discriminativa
    threshold_map = {
        "titanet": TITANET_MATCH_THRESHOLD,   # 0.7  — 192 dims, muy discriminativo
        "pyannote": EMBEDDING_MATCH_THRESHOLD, # 0.75 — 256 dims
        "mfcc": MFCC_MATCH_THRESHOLD,          # 0.65 — 40 dims, menos discriminativo
    }
    threshold = threshold_map.get(emb_type, EMBEDDING_MATCH_THRESHOLD)
    speaker_map = {}

    # Evitar asignar el mismo MC a dos speakers distintos
    used_names = set()

    # Filtrar voicedb: solo comparar contra MCs del mismo tipo de embedding.
    # Si la DB tiene "Chuty" con titanet (192d) y "Wos" con pyannote (256d),
    # comparar un embedding titanet contra el pyannote da 0.0 (dims distintas)
    # y confunde la identificación.
    compatible_mcs = {
        name: data for name, data in voicedb.items()
        if data.get("type", "?") == emb_type
    }
    if len(compatible_mcs) < len(voicedb):
        n_skip = len(voicedb) - len(compatible_mcs)
        print(f"  ℹ️  {n_skip} MC(s) en DB con tipo distinto a '{emb_type}' (ignorados)")

    for spk_label, emb in embeddings.items():
        best_name = None
        best_sim = -1

        for mc_name, mc_data in compatible_mcs.items():
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
                print(f"  ❓ {spk_label} — no hay MCs compatibles en la DB (tipo={emb_type})")

    return speaker_map

def register_new_speakers(embeddings, speaker_map, voicedb, emb_type="pyannote"):
    """Pregunta al usuario si quiere registrar speakers no identificados.

    Para cada speaker marcado como "(Nuevo)", ofrece al usuario la opción
    de darle un nombre. Si el MC ya existía en la DB, promedia el embedding
    viejo con el nuevo (media móvil) para robustecer el reconocimiento.
    Guarda el tipo de embedding ("titanet", "pyannote" o "mfcc") para no
    mezclar dimensiones distintas en futuras comparaciones.

    Returns:
        voicedb actualizada (también se guarda a disco)
    """
    nuevos = {label: name for label, name in speaker_map.items()
              if name.endswith("(Nuevo)")}

    if not nuevos:
        return voicedb

    print("\n" + "="*60)
    print("  📋 REGISTRO DE MCs")
    print("="*60)
    print("  Hay speakers no identificados. ¿Quieres registrarlos?")
    print("  (Deja vacío para saltar)")
    print("  Nota: Si hay un Host/Presentador, etiquetalo como 'Host'")
    print("        y no como MC, ya que la IA separa todas las voces unicas.\n")

    for spk_label in nuevos:
        if spk_label not in embeddings:
            continue
        nombre = input(f"  Nombre para {spk_label}: ").strip()
        if not nombre:
            continue

        emb = embeddings[spk_label]
        # Buscar match case-insensitive para evitar duplicados "Chuty"/"chuty"
        nombre_real = nombre
        for existing_name in voicedb:
            if existing_name.lower() == nombre.lower():
                nombre_real = existing_name
                break
        nombre = nombre_real

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
# DIARIZACIÓN NVIDIA (SORTFORMER v2 + TITANET)
# ============================================================================

def _extract_titanet_embeddings(audio_path, diarize_df, device):
    """Extrae embeddings de TitaNet (192 dims) para cada speaker detectado.

    TitaNet es un modelo de verificación de speaker que genera un vector
    compacto (192 dims) que captura la identidad vocal. A diferencia de
    pyannote (256 dims), es más robusto a ruido y variaciones de grabación.

    Para cada speaker: concatena sus segmentos de audio (máx 30s para
    estabilidad del embedding), escribe un wav temporal y extrae el vector.

    Args:
        audio_path: ruta al archivo de audio original
        diarize_df: DataFrame con columnas start, end, speaker
        device: "cuda" o "cpu"

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

    # Cargar audio completo a 16kHz (lo que TitaNet espera)
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    embeddings = {}
    speakers = diarize_df["speaker"].unique()

    for spk in speakers:
        # Filtrar segmentos de este speaker
        spk_segs = diarize_df[diarize_df["speaker"] == spk]

        # Concatenar audio del speaker (máx 30s para estabilidad)
        chunks = []
        total_dur = 0.0
        max_dur = 30.0
        for _, row in spk_segs.iterrows():
            start_sample = int(row["start"] * sr)
            end_sample = int(row["end"] * sr)
            chunk = y[start_sample:end_sample]
            chunk_dur = len(chunk) / sr
            if total_dur + chunk_dur > max_dur:
                # Tomar solo lo que falta para llegar a 30s
                remaining = max_dur - total_dur
                chunk = chunk[:int(remaining * sr)]
                chunks.append(chunk)
                break
            chunks.append(chunk)
            total_dur += chunk_dur

        if not chunks:
            continue

        concatenated = np.concatenate(chunks)

        # Escribir wav temporal para TitaNet (requiere archivo en disco)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, concatenated, sr)

            # Extraer embedding usando infer_segment().
            # TitaNet no tiene un método get_embedding() directo.
            # verify_speakers() retorna un bool, no embeddings.
            # La forma correcta es pasar el audio por el encoder:
            with torch.no_grad():
                # Cargar audio como tensor para el modelo
                audio_tensor = torch.tensor(concatenated, dtype=torch.float32).unsqueeze(0).to(device)
                audio_len = torch.tensor([len(concatenated)], dtype=torch.long).to(device)

                # Forward pass por el encoder → embedding
                # TitaNet hereda de EncDecSpeakerLabelModel que tiene
                # forward() retornando (logits, embs)
                logits, emb_tensor = titanet.forward(
                    input_signal=audio_tensor,
                    input_signal_length=audio_len
                )
                emb_np = emb_tensor.squeeze().cpu().numpy()
                embeddings[spk] = emb_np.tolist()
                print(f"    ✅ {spk}: embedding {len(embeddings[spk])} dims")
        except Exception as e:
            print(f"    ⚠️  Error extrayendo embedding de {spk}: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return embeddings


def try_diarize_sortformer(audio_path, min_speakers=2):
    """Diarización con NVIDIA Sortformer v2 (DER ~6.57% para 2 speakers).

    Sortformer v2 es un modelo end-to-end que detecta quién habla cuándo
    sin necesidad de clustering externo. Es ~3x más preciso que pyannote
    para batallas de 2 MCs.

    Importante: Sortformer NO genera embeddings de speaker, por eso se
    complementa con TitaNet para la identificación de MCs.

    Args:
        audio_path: ruta al archivo de audio
        min_speakers: número mínimo de speakers esperados

    Returns:
        (diarize_df, embeddings) donde:
        - diarize_df: pd.DataFrame con columnas start, end, speaker
          (formato compatible con whisperx.assign_word_speakers)
        - embeddings: dict {speaker_label: list[float]} de TitaNet (192 dims)
        Retorna (None, {}) si falla
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

        # NeMo espera audio mono (batch, time). Si el archivo es estéreo,
        # lo convertimos a mono en un WAV temporal antes de pasarlo al modelo.
        # Usamos soundfile para leer los metadatos sin cargar todo en memoria.
        info = sf.info(audio_path)
        if info.channels > 1:
            print(f"  🔄 Audio estéreo detectado ({info.channels}ch) — convirtiendo a mono...")
            y_raw, sr_raw = librosa.load(audio_path, sr=None, mono=True)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_mono_path = tmp.name
            tmp.close()
            sf.write(tmp_mono_path, y_raw, sr_raw)
            diarize_path = tmp_mono_path
        else:
            diarize_path = audio_path

        print(f"  ⏳ Cargando Sortformer v2 en {device}...")
        sortformer = SortformerEncLabelModel.from_pretrained(SORTFORMER_MODEL_NAME)
        sortformer = sortformer.to(device)
        sortformer.eval()

        print(f"  ⏳ Diarizando con Sortformer v2...")
        # Sortformer v2 es end-to-end — no usa clustering tradicional,
        # así que no acepta num_speakers como parámetro directo.
        # Detecta automáticamente cuántos speakers hay (hasta 4).
        annotations = sortformer.diarize(
            audio=[diarize_path],
            batch_size=1,
            num_workers=0,
        )

        # Parsear output de Sortformer.
        # .diarize() retorna una lista (por archivo) de listas de strings.
        # Cada string tiene formato: "start end speaker_label"
        # Ejemplo: "0.12 0.89 speaker_0"
        import re
        rttm_pattern = re.compile(r"(\d+\.?\d*)\s+(\d+\.?\d*)\s+(speaker_\d+)", re.IGNORECASE)

        rows = []
        if annotations and len(annotations) > 0:
            for seg in annotations[0]:
                if isinstance(seg, str):
                    # Formato string: "start end speaker_X"
                    match = rttm_pattern.match(seg.strip())
                    if match:
                        start, end, speaker = match.groups()
                        rows.append({
                            "start": float(start),
                            "end": float(end),
                            "speaker": speaker
                        })
                elif isinstance(seg, tuple) and len(seg) >= 3:
                    rows.append({
                        "start": float(seg[0]),
                        "end": float(seg[1]),
                        "speaker": str(seg[2])
                    })
                elif hasattr(seg, 'start'):
                    rows.append({
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "speaker": str(getattr(seg, 'speaker', getattr(seg, 'label', 'unknown')))
                    })
                else:
                    # Formato desconocido — logear para debug
                    print(f"    [DEBUG] Segmento no reconocido: type={type(seg).__name__}, repr={repr(seg)[:100]}")

        if not rows:
            print("  ⚠️  Sortformer no detectó segmentos")
            return None, {}

        diarize_df = pd.DataFrame(rows)

        # Normalizar labels a formato SPEAKER_XX para consistencia
        unique_speakers = sorted(diarize_df["speaker"].unique())
        label_map = {old: f"SPEAKER_{i:02d}" for i, old in enumerate(unique_speakers)}
        diarize_df["speaker"] = diarize_df["speaker"].map(label_map)

        n_speakers = len(unique_speakers)
        print(f"  ✅ Sortformer v2: {n_speakers} speakers, {len(rows)} segmentos")

        # Extraer embeddings con TitaNet para identificación de MCs
        embeddings = _extract_titanet_embeddings(audio_path, diarize_df, device)

        return diarize_df, embeddings

    except Exception as e:
        print(f"  ⚠️  Sortformer v2 falló: {e}")
        import traceback
        traceback.print_exc()
        return None, {}
    finally:
        # Limpiar archivo mono temporal si se creó
        if tmp_mono_path and os.path.exists(tmp_mono_path):
            os.unlink(tmp_mono_path)


# ============================================================================
# TRANSCRIPCIÓN HÍBRIDA (WHISPERX / WHISPER)
# ============================================================================

def _chunked_transcribe(model, audio, chunk_s=90, overlap_s=15):
    """Transcribe audio largo en ventanas solapadas.

    PROBLEMA ORIGINAL: WhisperX usa Silero VAD para detectar habla. Cuando
    hay una pausa larga con ruido de público (9-10s de gritos entre patrones),
    el VAD "se rinde" y no reactiva la detección → se pierden los últimos
    patrones de la batalla.

    SOLUCIÓN: Partir el audio en chunks solapados (ej. 90s + 15s overlap)
    y transcribir cada uno independientemente. Así:
    - Los 10s de gritos quedan en el MEDIO de algún chunk, no al final
    - Cada chunk tiene su propio pase de VAD desde cero
    - No hay "contaminación" de ruido previo

    Deduplicación: en las zonas de overlap, puede haber segmentos
    duplicados. Se eliminan comparando timestamps (< 1.5s de diferencia
    en start = probable duplicado, se queda el de mayor contenido).

    Args:
        model: modelo WhisperX ya cargado
        audio: numpy array del audio (16kHz, como retorna whisperx.load_audio)
        chunk_s: duración de cada chunk en segundos
        overlap_s: solapamiento entre chunks consecutivos

    Returns:
        lista de segments (dicts con start, end, text)
    """
    sr = 16000  # WhisperX siempre trabaja a 16kHz
    total_samples = len(audio)
    total_duration = total_samples / sr

    # Si el audio es corto, transcribir de una sola pasada
    if total_duration <= chunk_s + 10:
        result = model.transcribe(audio, batch_size=16, language="es")
        return result.get("segments", [])

    chunk_samples = int(chunk_s * sr)
    overlap_samples = int(overlap_s * sr)
    step_samples = chunk_samples - overlap_samples

    all_segments = []
    chunk_idx = 0
    pos = 0

    print(f"  📦 Transcripción por chunks ({total_duration:.0f}s de audio, chunks de {chunk_s}s)")

    while pos < total_samples:
        end = min(pos + chunk_samples, total_samples)
        chunk = audio[pos:end]

        chunk_start_s = pos / sr
        chunk_end_s = end / sr
        chunk_dur = (end - pos) / sr

        # No procesar chunks muy cortos (< 5s restantes)
        if chunk_dur < 5:
            break

        print(f"    Chunk {chunk_idx}: {chunk_start_s:.0f}s → {chunk_end_s:.0f}s ({chunk_dur:.0f}s)")

        result = model.transcribe(chunk, batch_size=16, language="es")

        for seg in result.get("segments", []):
            # Ajustar timestamps a posición absoluta en el audio completo
            seg["start"] = seg.get("start", 0) + chunk_start_s
            seg["end"] = seg.get("end", 0) + chunk_start_s
            all_segments.append(seg)

        chunk_idx += 1
        pos += step_samples

    # Deduplicar segmentos en zonas de overlap:
    # Si dos segmentos tienen start dentro de 1.5s, son probables duplicados
    if all_segments:
        all_segments.sort(key=lambda s: s.get("start", 0))
        deduped = [all_segments[0]]
        for seg in all_segments[1:]:
            prev = deduped[-1]
            time_diff = abs(seg.get("start", 0) - prev.get("start", 0))
            if time_diff < 1.5:
                # Quedarse con el que tiene más texto (más contenido real)
                if len(seg.get("text", "")) > len(prev.get("text", "")):
                    deduped[-1] = seg
            else:
                deduped.append(seg)
        n_dupes = len(all_segments) - len(deduped)
        if n_dupes > 0:
            print(f"    Dedup: {n_dupes} segmentos duplicados eliminados de zonas de overlap")
        all_segments = deduped

    print(f"  ✅ {len(all_segments)} segmentos totales de {chunk_idx} chunks")
    return all_segments


def try_transcribe_whisperx(path):
    """Transcripción con WhisperX + diarización con cadena de fallback.

    La transcripción siempre usa WhisperX. Para diarización, se intenta en orden:
    1. Sortformer v2 + TitaNet (si NeMo disponible) → DER ~6.57%, embeddings 192 dims
    2. pyannote (si whisperx.diarize disponible) → DER ~19.9%, embeddings 256 dims
    3. Ninguno → retorna segments sin diarizar, main() aplica fallback espectral

    Returns:
        3-tupla (segments, embeddings, emb_type) donde:
        - segments: lista de dicts con text, start, end, speaker
        - embeddings: dict {speaker_label: list[float]}
        - emb_type: "titanet" | "pyannote" | "whisperx_no_diarize" | "failed"
    """
    print(f"\n📝 [PASO 5] Intentando WhisperX...")
    try:
        import torch
        import whisperx
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Transcripción (siempre WhisperX) ---
        print(f"  ⏳ Cargando modelo {WHISPERX_MODEL} en {device}...")
        model = whisperx.load_model(
            WHISPERX_MODEL, device,
            compute_type=COMPUTE_TYPE,
            language="es",
        )
        audio = whisperx.load_audio(path)

        # Transcripción por chunks solapados: evita que el VAD "se rinda"
        # cuando hay gritos de público entre patrones. Cada chunk de 90s
        # tiene su propio pase de VAD, así los 10s de ruido quedan en
        # el MEDIO de un chunk (no al final) y no matan la transcripción.
        print(f"  ⏳ Transcribiendo (idioma: español)...")
        segments = _chunked_transcribe(model, audio, chunk_s=90, overlap_s=15)
        result = {"segments": segments, "language": "es"}

        # Align
        print(f"  ⏳ Alineando...")
        model_a, meta = whisperx.load_align_model(language_code="es", device=device)
        result = whisperx.align(result["segments"], model_a, meta, audio, device, return_char_alignments=False)

        # --- Diarización: cadena de fallback ---
        diarize_df = None
        embeddings = {}
        emb_type = "whisperx_no_diarize"

        # [1] Intentar Sortformer v2 + TitaNet (más preciso)
        if BACKEND_SORTFORMER:
            print(f"  ⏳ Diarizando (Sortformer v2 + TitaNet)...")
            diarize_df, embeddings = try_diarize_sortformer(path, min_speakers=2)
            if diarize_df is not None:
                emb_type = "titanet"

        # [2] Fallback: pyannote vía WhisperX
        if diarize_df is None and BACKEND_PYANNOTE:
            print(f"  ⏳ Diarizando (pyannote)...")
            try:
                from whisperx.diarize import DiarizationPipeline
                diarize_model = DiarizationPipeline(token=HF_TOKEN, device=device)
                # min_speakers=3 para detectar host + 2 MCs. Sin max_speakers
                # para que pyannote decida si hay más (público, etc.)
                diarize_result = diarize_model(audio, min_speakers=3, return_embeddings=True)

                # Cuando return_embeddings=True, retorna tupla
                # (diarize_df, {speaker_label: embedding_vector})
                if isinstance(diarize_result, tuple):
                    diarize_df, embeddings = diarize_result
                    embeddings = {k: v.tolist() if hasattr(v, 'tolist') else list(v)
                                  for k, v in embeddings.items()}
                else:
                    diarize_df = diarize_result
                emb_type = "pyannote"
            except Exception as e:
                err_str = str(e)
                print(f"  ⚠️  pyannote falló: {err_str[:120]}")
                if "403" in err_str or "gated" in err_str:
                    print("  ⚠️  Tu token HF no tiene acceso al modelo de diarización.")
                    print("  ⚠️  Acepta la licencia en: https://hf.co/pyannote/speaker-diarization-community-1")
                    print("  ⚠️  Y también en: https://hf.co/pyannote/segmentation-3.0")

        # [3] Si no hay diarización disponible, retornar sin diarizar
        if diarize_df is None:
            print(f"  ⚠️  Sin diarización disponible — main() aplicará fallback espectral")
            return result["segments"], {}, "whisperx_no_diarize"

        # Asignar speakers a palabras usando el DataFrame de diarización
        result = whisperx.assign_word_speakers(diarize_df, result)

        print(f"  ✅ WhisperX + diarización ({emb_type}) éxito.")
        return result["segments"], embeddings, emb_type

    except Exception as e:
        err_str = str(e)
        print(f"  ⚠️  WhisperX falló: {err_str[:120]}")
        if "403" not in err_str and "gated" not in err_str:
            print("  ⚠️  (Probablemente incompatibilidad de torchaudio/Windows/Python 3.13)")
        print("  🔄 Cambiando a FALLBACK: Whisper + Diarización Espectral...")
        return None, {}, "failed"

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
    """Fallback: Whisper estándar + diarización espectral MFCC.

    Returns:
        3-tupla (segments, embeddings, emb_type) — emb_type es "mfcc" o "failed"
    """
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

        return segs, mfcc_embeddings, "mfcc"
    except Exception as e:
        print(f"  ❌ Fallback también falló: {e}")
        import traceback
        traceback.print_exc()
        return [], {}, "failed"

def save_txt(segments, input_name, method):
    with open(OUTPUT_TRANSCRIPT_PATH, "w", encoding="utf-8") as f:
        f.write(f"FlowMetrics v0.7 — {method}\n")
        f.write(f"Source: {input_name}\n\n")
        for s in segments:
            spk = s.get("speaker", "UNKNOWN")
            text = s.get("text", "").strip()
            # Solo omitir segmentos sin texto real (vacíos o whitespace)
            # NUNCA omitir por speaker — incluso "UNKNOWN" puede contener
            # contenido válido de batalla (host introduce ronda, etc.)
            if not text:
                continue
            f.write(f"[{spk}] [{int(s['start']//60):02d}:{s['start']%60:04.1f}] {text}\n")
    print(f"  📄 Saved: {OUTPUT_TRANSCRIPT_PATH}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    _patch_torchaudio() # Patch init
    
    print("="*60 + "\n  🎤 FlowMetrics v0.7 (Sortformer v2 / WhisperX / Spectral)\n" + "="*60)
    
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

    # 3. Transcribe (Hybrid) — retorna 3-tupla (segments, embeddings, emb_type)
    segments, embeddings, emb_type = try_transcribe_whisperx(vox)

    if segments is None:
        # WhisperX falló completamente → fallback a Whisper + espectral
        segments, embeddings, emb_type = transcribe_fallback_whisper(vox)
    elif emb_type == "whisperx_no_diarize":
        # WhisperX transcribió pero no pudo diarizar → aplicar fallback espectral
        print("  🔬 Aplicando diarización espectral como fallback...")
        segments, mfcc_embeddings = _spectral_diarize(vox, segments, num_speakers=2)
        embeddings = mfcc_embeddings
        emb_type = "mfcc"

    # Seleccionar nombre del método para la transcripción
    method_names = {
        "titanet": "WhisperX + Sortformer v2 + TitaNet",
        "pyannote": "WhisperX + pyannote (Diarized)",
        "mfcc": "Whisper + Spectral Diarization (Fallback)",
        "failed": "Whisper (Sin Diarización)",
    }
    method = method_names.get(emb_type, f"WhisperX ({emb_type})")

    # Aviso de migración: si voicedb tiene embeddings de un tipo distinto
    # al que se está usando ahora, las dimensiones son incompatibles
    if voicedb and embeddings:
        db_types = {mc_data.get("type", "?") for mc_data in voicedb.values()}
        if emb_type not in db_types and db_types != {"?"}:
            print(f"\n  ⚠️  MIGRACIÓN: La VoiceDB tiene embeddings tipo {db_types}")
            print(f"  ⚠️  pero ahora se usa '{emb_type}' (dims incompatibles).")
            print(f"  ⚠️  Los MCs se deberán re-registrar con el nuevo tipo.")

    # 4. Identificar speakers por huella vocal
    speaker_map = {}
    if embeddings:
        print(f"\n🔍 Identificando speakers ({emb_type})...")
        speaker_map = identify_speakers(embeddings, voicedb, emb_type=emb_type)
        segments = rename_segments(segments, speaker_map)

    save_txt(segments, os.path.basename(infile), method)

    # 4b. Diagnóstico de cobertura — avisar si la transcripción no cubre todo
    if segments:
        audio_dur = librosa.get_duration(y=y_vox, sr=sr_vox)
        last_seg_end = max(s.get("end", 0) for s in segments)
        cobertura_pct = (last_seg_end / audio_dur * 100) if audio_dur > 0 else 0
        if cobertura_pct < 85:
            print(f"\n  ⚠️  Cobertura: {cobertura_pct:.0f}% del audio transcrito ({last_seg_end:.0f}s de {audio_dur:.0f}s)")
            print(f"      Si faltan patrones, puede ser por pausas con ruido de")
            print(f"      publico que Whisper no detecta como habla. Tip: recorta")
            print(f"      el audio solo a las rondas de MC para mejor resultado.")

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

    # 5b. Separar speakers "menores" del scoreboard (NO eliminar de embeddings)
    # En batallas hay MCs + host + público. Los speakers con muy pocas
    # sílabas respecto al total probablemente son host/público, pero
    # NO los borramos de embeddings (el usuario decide al registrar)
    # y NO los borramos de la transcripción (puede haber contenido válido).
    # Solo se excluyen del SCOREBOARD de métricas.
    total_all_syl = sum(
        calc_metrics(np.array(o), grid)[4]  # [4] = total sílabas
        for o in speaker_onsets.values() if o
    )
    # Umbral relativo: speaker con < 2% del total de sílabas O < 5 absolutas
    # PERO: nunca filtrar si quedarían menos de 2 speakers en el scoreboard
    speakers_para_score = {}
    speakers_menores = {}
    for spk, ons in speaker_onsets.items():
        if not ons:
            continue
        msp_chk, _, _, _, tot_chk = calc_metrics(np.array(ons), grid)
        pct_of_total = (tot_chk / total_all_syl * 100) if total_all_syl > 0 else 0
        if tot_chk < 5 or (pct_of_total < 2.0 and msp_chk == 0):
            speakers_menores[spk] = (tot_chk, pct_of_total)
        else:
            speakers_para_score[spk] = ons

    # Seguridad: si filtramos demasiado y quedan < 2, devolver los filtrados
    if len(speakers_para_score) < 2 and speakers_menores:
        # Reincorporar los menores ordenados por sílabas (de más a menos)
        for spk, (tot_chk, _) in sorted(speakers_menores.items(), key=lambda x: -x[1][0]):
            if len(speakers_para_score) >= 2:
                break
            speakers_para_score[spk] = speaker_onsets[spk]
            del speakers_menores[spk]  # Ya no es "menor"

    if speakers_menores:
        names = ", ".join(f"{s} ({n} sil, {p:.1f}%)" for s, (n, p) in speakers_menores.items())
        print(f"\n  🔇 Excluidos del scoreboard (Host/Publico): {names}")
        print(f"      (Siguen en la transcripcion y se pueden registrar en VoiceDB)")

    # 6. Biometría de Flow + Scoreboard dual
    def _bar(value, max_val=100, width=20):
        """Genera barra visual ASCII: [||||||||    ] 65/100"""
        filled = int((value / max_val) * width)
        return f"[{'|' * filled}{' ' * (width - filled)}]"

    print("\n" + "="*62 + "\n  🏆 SCOREBOARD — TÉCNICA vs GROOVE\n" + "="*62)

    # Recopilar resultados para determinar ganadores
    resultados = {}

    for spk, onsets in speakers_para_score.items():
        if not onsets: continue
        msp, pt, acc, h, tot = calc_metrics(np.array(onsets), grid)

        # Extraer audio del speaker para análisis DSP
        spk_audio = extract_speaker_audio(y_vox, sr_vox, segments, spk)

        # Análisis biométrico dual (técnica + groove)
        bio = analizar_biometria_flow(
            spk_audio, np.array(onsets), grid, sr_vox,
            sps_max=msp, total_silabas=tot
        )
        resultados[spk] = {"bio": bio, "msp": msp, "acc": acc, "tot": tot, "pt": pt}

        # Buscar la línea donde alcanzó el SPS máximo
        rhyme = ""
        for s in segments:
            if s.get("speaker") == spk and s['start'] <= pt <= s['end']+0.5:
                rhyme = s['text'].strip()
                break

        # --- Imprimir scoreboard dual ---
        t = bio["indice_tecnica"]
        g = bio["indice_groove"]
        print(f"\n  {'='*60}")
        print(f"  {spk}")
        print(f"  {'='*60}")
        print(f"  TECNICA  {_bar(t)}  {t:.1f}/100")
        print(f"  GROOVE   {_bar(g)}  {g:.1f}/100")
        print(f"  {'-'*60}")
        # Desglose Técnica
        print(f"  Tecnica: precision={bio['pts_precision']:.0f}"
              f" + velocidad={bio['pts_velocidad']:.0f}"
              f" + volumen={bio['pts_volumen']:.0f}")
        print(f"    SPS Max: {msp:<2.0f}  |  Acc: {acc:.1f}%  |  Silabas: {tot}")
        print(f"  {'-'*60}")
        # Desglose Groove
        print(f"  Groove:  swing={bio['pts_swing']:.0f}"
              f" + sinc={bio['pts_sincopa']:.0f}"
              f" + smooth={bio['pts_smooth']:.0f}"
              f" + sust={bio['pts_sustain']:.0f}"
              f" + din={bio['pts_dinamica']:.0f}")
        print(f"    Swing:    {bio['swing_label']:<13} (mu={bio['delta_media_ms']:.1f}ms, sigma={bio['delta_std_ms']:.1f}ms)")
        print(f"    Sincopa:  {bio['sincopa_label']:<13} ({bio['pct_sincopa']:.1f}% en contratiempos)")
        print(f"    Melodia:  {bio['pitch_label']:<13} (smooth={bio['smooth_score']:.0f}, voiced={bio['pct_voiced']:.0f}%, dF0={bio['median_delta_f0']:.1f}Hz)")
        print(f"    Sustain:  {bio['chicleadas']} chicleo(s)      ({bio['pct_sustain']:.1f}% del tiempo)")
        print(f"    Dinamica: {bio['dinamica_label']:<13} (var={bio['varianza_rms']:.4f})")
        if rhyme:
            print(f"    Peak:     \"{rhyme[:51]}...\"")

    # --- Determinar ganadores ---
    if len(resultados) >= 2:
        spks = list(resultados.keys())
        print(f"\n  {'='*60}")
        print(f"  VEREDICTO")
        print(f"  {'='*60}")

        # Ganador Técnica
        best_tec = max(spks, key=lambda s: resultados[s]["bio"]["indice_tecnica"])
        t_score = resultados[best_tec]["bio"]["indice_tecnica"]

        # Ganador Groove
        best_grv = max(spks, key=lambda s: resultados[s]["bio"]["indice_groove"])
        g_score = resultados[best_grv]["bio"]["indice_groove"]

        print(f"  Tecnica: {best_tec} ({t_score:.1f})")
        print(f"  Groove:  {best_grv} ({g_score:.1f})")

        if best_tec == best_grv:
            print(f"\n  {best_tec} domina ambos ejes.")
        else:
            print(f"\n  Estilos distintos — depende del criterio del jurado.")
        print()

    # 7. Registro de nuevos MCs (si hay embeddings disponibles)
    if embeddings:
        voicedb = register_new_speakers(embeddings, speaker_map, voicedb, emb_type=emb_type)
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
