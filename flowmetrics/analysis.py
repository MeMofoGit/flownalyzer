"""
Módulo de análisis DSP: beats, onsets, métricas y biometría de flow.

Contiene toda la lógica matemática para medir el flow de un MC:
- Detección de beats y generación de cuadrícula rítmica
- Detección de sílabas (onsets)
- Métricas básicas (SPS, precisión rítmica)
- Biometría avanzada (micro-timing, síncopa, pitch, sustain, dinámica)
- Índices duales: Técnica (0-100) y Groove (0-100)
"""

import numpy as np
import librosa

from .config import SR, BEAT_MATCH_THRESHOLD_S


# ============================================================================
# DETECCIÓN DE BEATS Y CUADRÍCULA
# ============================================================================

def detect_beats(y, sr):
    """Detecta tempo y posiciones de beats en la instrumental.

    Returns:
        (tempo_bpm, beat_times) — tempo float y array de tiempos en segundos
    """
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, 'item'):
        tempo = tempo.item()
    elif isinstance(tempo, np.ndarray):
        tempo = tempo[0]
    return float(tempo), librosa.frames_to_time(beats, sr=sr)


def generate_grid(tempo, beats, duration):
    """Genera cuadrícula de semicorcheas sincronizada con los beats.

    La cuadrícula se resincroniza con cada beat detectado para evitar
    deriva acumulada. Esto es importante porque el tempo real de una
    instrumental no es perfectamente constante.

    Returns:
        numpy array con tiempos de cada semicorchea
    """
    sub_dur = 60.0 / tempo / 4
    grid = []
    t = 0
    beat_idx = 0
    while t < duration:
        if beat_idx < len(beats) and abs(t - beats[beat_idx]) < sub_dur * 0.5:
            t = beats[beat_idx]
            beat_idx += 1
        grid.append(t)
        t += sub_dur
    return np.unique(np.array(grid))


# ============================================================================
# DETECCIÓN DE SÍLABAS
# ============================================================================

def detect_onsets(y, sr):
    """Detecta sílabas (onsets vocales) en la acapella.

    Usa onset_detect con backtrack para mayor precisión temporal.
    delta=0.07 balancea entre captar sílabas suaves y filtrar ruido.

    Returns:
        numpy array con tiempos de cada onset en segundos
    """
    onsets = librosa.onset.onset_detect(
        y=y, sr=sr, backtrack=True, wait=1, delta=0.07,
    )
    return librosa.frames_to_time(onsets, sr=sr)


# ============================================================================
# MÉTRICAS BÁSICAS
# ============================================================================

def calc_metrics(onsets, grid, window=1.0):
    """Calcula SPS máximo y precisión rítmica.

    Args:
        onsets: array de tiempos de onset
        grid: cuadrícula de semicorcheas
        window: ventana en segundos para SPS máximo

    Returns:
        dict con: sps_max, peak_time, accuracy_pct, hits, total_onsets
    """
    if len(onsets) == 0:
        return {"sps_max": 0, "peak_time": 0.0, "accuracy_pct": 0.0,
                "hits": 0, "total_onsets": 0}

    # SPS Max — ventana deslizante
    max_sps = 0
    peak_t = 0
    for t in onsets:
        c = np.searchsorted(onsets, t + window) - np.searchsorted(onsets, t)
        if c > max_sps:
            max_sps, peak_t = c, t

    # Beat Match — distancia al punto más cercano de la cuadrícula
    hits = _count_grid_hits(onsets, grid)
    acc = (hits / len(onsets)) * 100

    return {
        "sps_max": max_sps,
        "peak_time": float(peak_t),
        "accuracy_pct": round(acc, 1),
        "hits": hits,
        "total_onsets": len(onsets),
    }


def _count_grid_hits(onsets, grid):
    """Cuenta cuántos onsets caen dentro del umbral de la cuadrícula."""
    sorted_grid = np.sort(grid)
    hits = 0
    for t in onsets:
        idx = np.searchsorted(sorted_grid, t)
        d = float('inf')
        if idx < len(sorted_grid):
            d = min(d, abs(sorted_grid[idx] - t))
        if idx > 0:
            d = min(d, abs(sorted_grid[idx - 1] - t))
        if d <= BEAT_MATCH_THRESHOLD_S:
            hits += 1
    return hits


# ============================================================================
# UTILIDAD: EXTRAER AUDIO POR SPEAKER
# ============================================================================

def extract_speaker_audio(y_vox, sr, segments, speaker_label):
    """Extrae y concatena el audio de un speaker específico.

    Recorta los fragmentos correspondientes a los timestamps de la
    transcripción para el speaker indicado.

    Returns:
        numpy array mono con el audio concatenado del speaker
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


# ============================================================================
# BIOMETRÍA DE FLOW — SUB-ANÁLISIS
# ============================================================================

def _analizar_microtiming(onsets, grid):
    """Micro-timing / Índice de Swing.

    Para cada sílaba, calcula la diferencia en ms respecto a la
    semicorchea más cercana. La distribución revela el estilo:
    - σ bajo + μ 15-45ms → Laid-back (swing intencional)
    - σ bajo + μ < 15ms → Métrico (preciso)
    - σ > 50ms → Arrítmico

    Returns:
        dict con: delta_media_ms, delta_std_ms, bonus_swing, swing_label
    """
    if len(onsets) == 0 or len(grid) == 0:
        return {"delta_media_ms": 0.0, "delta_std_ms": 0.0,
                "bonus_swing": 0.0, "swing_label": "Sin datos"}

    sorted_grid = np.sort(grid)
    deltas_ms = []

    for t in onsets:
        idx = np.searchsorted(sorted_grid, t)
        d = float("inf")
        if idx < len(sorted_grid):
            d = min(d, abs(sorted_grid[idx] - t))
        if idx > 0:
            d = min(d, abs(sorted_grid[idx - 1] - t))
        deltas_ms.append(d * 1000.0)

    deltas = np.array(deltas_ms)
    mu = float(np.mean(deltas))
    sigma = float(np.std(deltas))

    bonus_swing = 0.0
    swing_label = "Neutro"

    if sigma < 25.0:
        if 15.0 <= mu <= 45.0:
            # Laid-back: retraso sistemático — recurso estilístico valorado
            # en hip-hop (Snoop Dogg, Biggie). Sweet spot ~30ms.
            distance_to_sweet = abs(mu - 30.0)
            bonus_swing = max(0, 8.0 - (distance_to_sweet / 15.0) * 8.0)
            swing_label = "Laid-back"
        elif mu < 15.0:
            bonus_swing = 3.0
            swing_label = "Metrico"
    elif sigma > 50.0:
        bonus_swing = -5.0
        swing_label = "Arritmico"
    else:
        swing_label = "Variable"

    return {
        "delta_media_ms": round(mu, 1),
        "delta_std_ms": round(sigma, 1),
        "bonus_swing": round(bonus_swing, 1),
        "swing_label": swing_label,
        "deltas_ms": deltas,  # Para visualización
    }


def _analizar_sincopa(onsets, grid):
    """Análisis de Síncopa (Contratiempos).

    En semicorcheas: posiciones 0,2 = fuertes, posiciones 1,3 = débiles.
    Un alto % en débiles = síncopa activa = groove.

    Returns:
        dict con: pct_sincopa, bonus_sincopa, sincopa_label
    """
    if len(onsets) == 0 or len(grid) < 4:
        return {"pct_sincopa": 0.0, "bonus_sincopa": 0.0,
                "sincopa_label": "Sin datos"}

    sorted_grid = np.sort(grid)
    en_debil = 0
    total_asignados = 0

    for t in onsets:
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

        if best_d > 0.060:
            continue

        total_asignados += 1
        posicion = best_idx % 4
        if posicion in (1, 3):
            en_debil += 1

    if total_asignados == 0:
        return {"pct_sincopa": 0.0, "bonus_sincopa": 0.0,
                "sincopa_label": "Sin datos"}

    pct_sincopa = (en_debil / total_asignados) * 100.0

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
    """Suavidad Melódica — mide intención musical real.

    Mide la DERIVADA de F0 (diferencia frame a frame):
    - MC que entona → derivada baja y continua (melódico)
    - MC que grita → saltos bruscos de frecuencia (agresivo)

    Usa mediana de |ΔF0| para ser robusto a outliers.

    Returns:
        dict con: smooth_score (0-100), pct_voiced, pitch_label,
                  rango_vocal, median_delta_f0
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
        frame_length=2048, hop_length=512,
    )

    voiced_mask = (f0 > 66) & (f0 < 299)
    valid_f0 = f0[voiced_mask]
    pct_voiced = (np.sum(voiced_mask) / len(f0)) * 100.0 if len(f0) > 0 else 0.0

    if len(valid_f0) < 20:
        return resultado_vacio

    p95 = float(np.percentile(valid_f0, 95))
    p5 = float(np.percentile(valid_f0, 5))
    rango_vocal = p95 - p5

    voiced_indices = np.where(voiced_mask)[0]
    consecutive_pairs = np.where(np.diff(voiced_indices) == 1)[0]

    if len(consecutive_pairs) < 10:
        return {
            "smooth_score": 0.0, "pct_voiced": round(pct_voiced, 1),
            "pitch_label": "Sin datos", "rango_vocal": round(rango_vocal, 1),
            "median_delta_f0": 0.0,
        }

    deltas_f0 = np.abs(np.diff(
        f0[voiced_indices[consecutive_pairs[0]:consecutive_pairs[-1] + 2]]
    ))
    median_delta = float(np.median(deltas_f0))

    # Mapear mediana de |ΔF0| a 0-100 (invertido):
    # ~1.5 Hz = muy suave/melódico, ~12 Hz = grita/percusivo
    if median_delta <= 1.5:
        smooth_score = 100.0
    elif median_delta >= 12.0:
        smooth_score = 0.0
    else:
        smooth_score = (1.0 - (median_delta - 1.5) / 10.5) * 100.0

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
        "f0_contour": valid_f0,  # Para visualización
    }


def _analizar_sustain_pct(speaker_audio, sr):
    """Sustain Real — % de tiempo en chicleo respecto al tiempo total.

    Mide el PORCENTAJE del tiempo que el MC pasa sosteniendo notas.
    Umbral al 30% del pico RMS para captar susurros sostenidos.

    Returns:
        dict con: pct_sustain, chicleadas, varianza_rms, dinamica_label
    """
    resultado_vacio = {
        "pct_sustain": 0.0, "chicleadas": 0,
        "varianza_rms": 0.0, "dinamica_label": "Sin datos",
    }

    if len(speaker_audio) < sr * 0.5:
        return resultado_vacio

    rms = librosa.feature.rms(
        y=speaker_audio, frame_length=2048, hop_length=512,
    )[0]

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

    pct_sustain = (frames_en_sustain / len(rms)) * 100.0 if len(rms) > 0 else 0.0

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
        "rms_contour": rms,  # Para visualización
    }


# ============================================================================
# BIOMETRÍA PRINCIPAL — ÍNDICES DUALES
# ============================================================================

def analizar_biometria_flow(speaker_audio, onsets, grid, sr,
                            sps_max=0, total_silabas=0):
    """Calcula los dos índices principales de flow.

    TÉCNICA (0-100): precisión rítmica (70%) + velocidad (20%) + volumen (10%)
    GROOVE (0-100): swing (25%) + síncopa (25%) + smooth pitch (25%)
                    + sustain (15%) + dinámica (10%)

    Returns:
        dict completo con sub-métricas y ambos índices
    """
    timing = _analizar_microtiming(onsets, grid)
    sincopa = _analizar_sincopa(onsets, grid)
    pitch = _analizar_smooth_pitch(speaker_audio, sr)
    sustain = _analizar_sustain_pct(speaker_audio, sr)

    # ═══════════════════════════════════════
    # ÍNDICE DE TÉCNICA (0-100)
    # ═══════════════════════════════════════

    # A) Precisión rítmica (0-70 pts)
    hits = _count_grid_hits(onsets, grid) if len(onsets) > 0 else 0
    accuracy_pct = (hits / len(onsets)) * 100.0 if len(onsets) > 0 else 0.0
    pts_precision = (accuracy_pct / 100.0) * 70.0

    # B) Velocidad — SPS normalizado (0-20 pts)
    # 15 SPS es el techo práctico en freestyle
    pts_velocidad = max(0.0, min(20.0, (sps_max - 5.0) / 10.0 * 20.0))

    # C) Volumen silábico (0-10 pts)
    pts_volumen = max(0.0, min(10.0, (total_silabas - 100) / 300.0 * 10.0))

    indice_tecnica = max(0.0, min(100.0,
        pts_precision + pts_velocidad + pts_volumen))

    # ═══════════════════════════════════════
    # ÍNDICE DE GROOVE (0-100)
    # ═══════════════════════════════════════

    # A) Swing consistente (0-25 pts)
    mu = timing["delta_media_ms"]
    sigma = timing["delta_std_ms"]
    if sigma < 20.0 and 20.0 <= mu <= 50.0:
        swing_quality = 1.0 - (abs(mu - 35.0) / 25.0)
        consistency_bonus = 1.0 - (sigma / 20.0)
        pts_swing = max(0.0, (swing_quality * 0.6 + consistency_bonus * 0.4) * 25.0)
    elif sigma < 25.0 and mu < 20.0:
        pts_swing = 8.0
    elif sigma > 50.0:
        pts_swing = 0.0
    else:
        pts_swing = max(0.0, 12.0 - sigma * 0.2)

    # B) Síncopa (0-25 pts)
    pct_s = sincopa["pct_sincopa"]
    if pct_s >= 65.0:
        pts_sincopa = 25.0
    elif pct_s >= 45.0:
        pts_sincopa = (pct_s - 45.0) / 20.0 * 25.0
    else:
        pts_sincopa = 0.0

    # C) Smooth Pitch (0-25 pts)
    if pitch["pct_voiced"] >= 30.0:
        pts_smooth = (pitch["smooth_score"] / 100.0) * 25.0
    else:
        pts_smooth = (pitch["smooth_score"] / 100.0) * 25.0 * (pitch["pct_voiced"] / 30.0)

    # D) Sustain Real (0-15 pts)
    pts_sustain = max(0.0, min(15.0, (sustain["pct_sustain"] / 30.0) * 15.0))

    # E) Dinámica de volumen (0-10 pts)
    var = sustain["varianza_rms"]
    if var >= 0.10:
        pts_dinamica = 10.0
    elif var >= 0.03:
        pts_dinamica = (var - 0.03) / 0.07 * 10.0
    else:
        pts_dinamica = 0.0

    indice_groove = max(0.0, min(100.0,
        pts_swing + pts_sincopa + pts_smooth + pts_sustain + pts_dinamica))

    return {
        # Micro-timing
        "delta_media_ms": timing["delta_media_ms"],
        "delta_std_ms": timing["delta_std_ms"],
        "swing_label": timing["swing_label"],
        "deltas_ms": timing.get("deltas_ms"),
        # Síncopa
        "pct_sincopa": sincopa["pct_sincopa"],
        "sincopa_label": sincopa["sincopa_label"],
        # Pitch
        "smooth_score": pitch["smooth_score"],
        "pct_voiced": pitch["pct_voiced"],
        "rango_vocal": pitch["rango_vocal"],
        "median_delta_f0": pitch["median_delta_f0"],
        "pitch_label": pitch["pitch_label"],
        "f0_contour": pitch.get("f0_contour"),
        # Sustain + Dinámica
        "pct_sustain": sustain["pct_sustain"],
        "chicleadas": sustain["chicleadas"],
        "varianza_rms": sustain["varianza_rms"],
        "dinamica_label": sustain["dinamica_label"],
        "rms_contour": sustain.get("rms_contour"),
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


def map_onsets_to_speakers(all_onsets, segments):
    """Asigna cada onset al speaker que estaba hablando en ese momento.

    Returns:
        dict {speaker_label: [onset_times]}
    """
    has_speakers = any(s.get("speaker", "UNKNOWN") != "UNKNOWN" for s in segments)
    speaker_onsets = {}

    if has_speakers:
        ranges = {}
        for s in segments:
            spk = s.get("speaker", "UNKNOWN")
            if spk not in ranges:
                ranges[spk] = []
            ranges[spk].append((s['start'], s['end']))

        for onset in all_onsets:
            assigned = False
            for spk, rngs in ranges.items():
                for (st, en) in rngs:
                    if st - 0.1 <= onset <= en + 0.1:
                        if spk not in speaker_onsets:
                            speaker_onsets[spk] = []
                        speaker_onsets[spk].append(onset)
                        assigned = True
                        break
                if assigned:
                    break
            if not assigned:
                speaker_onsets.setdefault("UNKNOWN", []).append(onset)
    else:
        speaker_onsets["UNKNOWN"] = list(all_onsets)

    return speaker_onsets


def filter_minor_speakers(speaker_onsets, grid):
    """Separa speakers principales (MCs) de menores (host/público).

    Speakers con < 2% del total de sílabas o < 5 absolutas se excluyen
    del scoreboard. Nunca deja menos de 2 speakers principales.

    Returns:
        (speakers_principales, speakers_menores)
        Cada uno es un dict {speaker: onsets_list}
    """
    total_all = sum(len(o) for o in speaker_onsets.values())
    principales = {}
    menores = {}

    for spk, ons in speaker_onsets.items():
        if not ons:
            continue
        m = calc_metrics(np.array(ons), grid)
        tot = m["total_onsets"]
        sps = m["sps_max"]
        pct = (tot / total_all * 100) if total_all > 0 else 0
        if tot < 5 or (pct < 2.0 and sps == 0):
            menores[spk] = ons
        else:
            principales[spk] = ons

    # Seguridad: si quedan < 2 principales, reincorporar menores
    if len(principales) < 2 and menores:
        for spk in sorted(menores, key=lambda s: -len(menores[s])):
            if len(principales) >= 2:
                break
            principales[spk] = menores.pop(spk)

    return principales, menores


def compute_rolling_sps(onsets, window=2.0, step=0.25):
    """Calcula SPS en ventana deslizante para timeline.

    Args:
        onsets: array de tiempos de onset
        window: duración de la ventana en segundos
        step: paso entre ventanas

    Returns:
        (times, sps_values) — arrays para plotear
    """
    if len(onsets) == 0:
        return np.array([]), np.array([])

    t_start = onsets[0]
    t_end = onsets[-1]
    times = np.arange(t_start, t_end - window + step, step)
    sps_values = []

    for t in times:
        count = np.sum((onsets >= t) & (onsets < t + window))
        sps_values.append(count / window)

    return times, np.array(sps_values)
