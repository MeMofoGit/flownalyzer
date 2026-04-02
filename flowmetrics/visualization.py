"""
Visualización profesional de métricas de flow.

Genera:
1. Dashboard multi-panel (PNG): radar chart, SPS timeline, waveform con onsets
2. Reporte HTML auto-contenido con gráficos embebidos y tablas de métricas
"""

import os
import io
import base64
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

from .config import SPEAKER_COLORS, EXTRA_COLORS
from .analysis import compute_rolling_sps


# ============================================================================
# PALETA DE COLORES
# ============================================================================

DARK_BG = '#0d1117'
PANEL_BG = '#161b22'
GRID_COLOR = '#30363d'
TEXT_COLOR = '#e6edf3'
TEXT_DIM = '#8b949e'
ACCENT = '#58a6ff'


def _get_speaker_color(speaker, index=0):
    """Retorna un color para el speaker, consistente entre gráficos."""
    if speaker in SPEAKER_COLORS:
        return SPEAKER_COLORS[speaker]
    return EXTRA_COLORS[index % len(EXTRA_COLORS)]


def _fig_to_base64(fig):
    """Convierte una figura matplotlib a string base64 para HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return b64


# ============================================================================
# RADAR CHART — COMPARACIÓN MULTI-DIMENSIONAL
# ============================================================================

def _draw_radar(ax, resultados):
    """Dibuja radar chart comparando speakers en 6 ejes.

    Ejes (normalizados a 0-100):
    - Precisión: accuracy_pct
    - Velocidad: SPS max (normalizado, 15 SPS = 100)
    - Melodía: smooth_score
    - Swing: pts_swing * 4 (normalizado de 0-25 a 0-100)
    - Síncopa: pct_sincopa
    - Dinámica: varianza_rms * 1000 (normalizado)
    """
    categories = ['Precisión', 'Velocidad', 'Melodía',
                  'Swing', 'Síncopa', 'Dinámica']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_facecolor(PANEL_BG)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=TEXT_COLOR, fontsize=9, fontweight='bold')

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'],
                        color=TEXT_DIM, fontsize=7)
    ax.tick_params(colors=GRID_COLOR)
    ax.spines['polar'].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.3)

    for idx, (spk, data) in enumerate(resultados.items()):
        bio = data["bio"]
        values = [
            bio["accuracy_pct"],
            min(100, (data["msp"] / 15.0) * 100),
            bio["smooth_score"],
            bio["pts_swing"] * 4,   # 0-25 → 0-100
            bio["pct_sincopa"],
            min(100, bio["varianza_rms"] * 1000),
        ]
        values += values[:1]

        color = _get_speaker_color(spk, idx)
        ax.plot(angles, values, 'o-', linewidth=2, color=color,
                label=spk, markersize=4)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
              fontsize=9, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR)


# ============================================================================
# SPS TIMELINE — VELOCIDAD A LO LARGO DE LA BATALLA
# ============================================================================

def _draw_sps_timeline(ax, speaker_onsets, resultados):
    """Dibuja la velocidad (SPS) de cada speaker a lo largo del tiempo."""
    ax.set_facecolor(PANEL_BG)

    for idx, (spk, onsets) in enumerate(speaker_onsets.items()):
        if spk not in resultados:
            continue
        onsets_arr = np.array(onsets)
        times, sps = compute_rolling_sps(onsets_arr, window=2.0, step=0.5)
        if len(times) == 0:
            continue

        color = _get_speaker_color(spk, idx)
        ax.plot(times, sps, color=color, linewidth=1.5, alpha=0.9, label=spk)
        ax.fill_between(times, sps, alpha=0.1, color=color)

    ax.set_xlabel('Tiempo (s)', color=TEXT_DIM, fontsize=9)
    ax.set_ylabel('Sílabas/s', color=TEXT_DIM, fontsize=9)
    ax.set_title('SPS Timeline (ventana 2s)', color=TEXT_COLOR,
                 fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color=GRID_COLOR, alpha=0.3)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR)


# ============================================================================
# WAVEFORM CON ONSETS Y BEATS
# ============================================================================

def _draw_waveform(ax, y_vox, sr, beats, speaker_onsets, resultados):
    """Waveform con beats (rojo) y onsets coloreados por speaker."""
    ax.set_facecolor(PANEL_BG)
    t = np.linspace(0, len(y_vox) / sr, len(y_vox))
    ax.plot(t, y_vox, color=TEXT_DIM, alpha=0.25, linewidth=0.3)

    for b in beats:
        ax.axvline(b, color='#ff6b6b', alpha=0.3, linewidth=0.5, linestyle='--')

    for idx, (spk, onsets) in enumerate(speaker_onsets.items()):
        if spk not in resultados:
            continue
        color = _get_speaker_color(spk, idx)
        ax.vlines(onsets, -0.7, 0.7, color=color, alpha=0.7,
                  linewidth=0.8, label=spk)

    ax.set_xlabel('Tiempo (s)', color=TEXT_DIM, fontsize=9)
    ax.set_title('Waveform + Onsets + Beats', color=TEXT_COLOR,
                 fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR, loc='upper right')


# ============================================================================
# SCOREBOARD BARS — TÉCNICA VS GROOVE
# ============================================================================

def _draw_scoreboard(ax, resultados):
    """Barras horizontales comparando Técnica y Groove por speaker."""
    ax.set_facecolor(PANEL_BG)

    speakers = list(resultados.keys())
    n = len(speakers)
    y_pos = np.arange(n)
    bar_height = 0.35

    for i, spk in enumerate(speakers):
        bio = resultados[spk]["bio"]
        color = _get_speaker_color(spk, i)

        # Técnica
        ax.barh(i + bar_height / 2, bio["indice_tecnica"], bar_height,
                color=color, alpha=0.9, label='Técnica' if i == 0 else '')
        ax.text(bio["indice_tecnica"] + 1, i + bar_height / 2,
                f'{bio["indice_tecnica"]:.1f}', va='center',
                color=TEXT_COLOR, fontsize=9, fontweight='bold')

        # Groove (más claro)
        ax.barh(i - bar_height / 2, bio["indice_groove"], bar_height,
                color=color, alpha=0.5, label='Groove' if i == 0 else '')
        ax.text(bio["indice_groove"] + 1, i - bar_height / 2,
                f'{bio["indice_groove"]:.1f}', va='center',
                color=TEXT_DIM, fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(speakers, color=TEXT_COLOR, fontsize=10, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.set_xlabel('Score', color=TEXT_DIM, fontsize=9)
    ax.set_title('Técnica (sólido) vs Groove (transparente)',
                 color=TEXT_COLOR, fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', color=GRID_COLOR, alpha=0.3)
    ax.invert_yaxis()


# ============================================================================
# MICRO-TIMING HISTOGRAM
# ============================================================================

def _draw_timing_histogram(ax, resultados):
    """Histograma de micro-timing (deltas vs grid) por speaker."""
    ax.set_facecolor(PANEL_BG)

    for idx, (spk, data) in enumerate(resultados.items()):
        deltas = data["bio"].get("deltas_ms")
        if deltas is None or len(deltas) == 0:
            continue
        color = _get_speaker_color(spk, idx)
        ax.hist(deltas, bins=40, range=(0, 80), alpha=0.5,
                color=color, label=f'{spk} (μ={data["bio"]["delta_media_ms"]:.0f}ms)',
                edgecolor=color, linewidth=0.5)

    ax.axvline(30, color='#f0883e', linestyle='--', alpha=0.7,
               linewidth=1, label='Sweet spot (30ms)')
    ax.set_xlabel('Delta vs Grid (ms)', color=TEXT_DIM, fontsize=9)
    ax.set_ylabel('Frecuencia', color=TEXT_DIM, fontsize=9)
    ax.set_title('Distribución de Micro-timing', color=TEXT_COLOR,
                 fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_COLOR)


# ============================================================================
# DASHBOARD PRINCIPAL (PNG)
# ============================================================================

def generate_dashboard(y_vox, sr, beats, speaker_onsets, resultados,
                       output_path):
    """Genera dashboard multi-panel PNG.

    Layout 3x2:
    [Radar Chart      ] [Scoreboard Bars  ]
    [SPS Timeline     ] [Timing Histogram ]
    [Waveform + Onsets                    ]
    """
    fig = plt.figure(figsize=(20, 16), facecolor=DARK_BG)

    # Título principal
    fig.suptitle('FlowMetrics v1.0 — Dashboard',
                 color=TEXT_COLOR, fontsize=18, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.94, top=0.93, bottom=0.05)

    # Fila 1
    ax_radar = fig.add_subplot(gs[0, 0], polar=True)
    _draw_radar(ax_radar, resultados)

    ax_score = fig.add_subplot(gs[0, 1])
    _draw_scoreboard(ax_score, resultados)

    # Fila 2
    ax_sps = fig.add_subplot(gs[1, 0])
    _draw_sps_timeline(ax_sps, speaker_onsets, resultados)

    ax_timing = fig.add_subplot(gs[1, 1])
    _draw_timing_histogram(ax_timing, resultados)

    # Fila 3 (span completo)
    ax_wave = fig.add_subplot(gs[2, :])
    _draw_waveform(ax_wave, y_vox, sr, beats, speaker_onsets, resultados)

    plt.savefig(output_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  ✅ Dashboard: {output_path}")
    return output_path


# ============================================================================
# REPORTE HTML AUTO-CONTENIDO
# ============================================================================

def generate_html_report(resultados, segments, speaker_onsets,
                         y_vox, sr, beats, tempo, input_name,
                         method, output_path):
    """Genera reporte HTML con gráficos embebidos y tablas de métricas.

    El HTML es auto-contenido (imágenes en base64, CSS inline) para
    compartir como archivo único.
    """
    # --- Generar gráficos como base64 ---
    # Radar
    fig_radar = plt.figure(figsize=(8, 6), facecolor=DARK_BG)
    ax = fig_radar.add_subplot(111, polar=True)
    _draw_radar(ax, resultados)
    radar_b64 = _fig_to_base64(fig_radar)
    plt.close(fig_radar)

    # Scoreboard
    fig_score = plt.figure(figsize=(8, 4), facecolor=DARK_BG)
    ax = fig_score.add_subplot(111)
    _draw_scoreboard(ax, resultados)
    score_b64 = _fig_to_base64(fig_score)
    plt.close(fig_score)

    # SPS Timeline
    fig_sps = plt.figure(figsize=(12, 4), facecolor=DARK_BG)
    ax = fig_sps.add_subplot(111)
    _draw_sps_timeline(ax, speaker_onsets, resultados)
    sps_b64 = _fig_to_base64(fig_sps)
    plt.close(fig_sps)

    # Timing histogram
    fig_timing = plt.figure(figsize=(8, 4), facecolor=DARK_BG)
    ax = fig_timing.add_subplot(111)
    _draw_timing_histogram(ax, resultados)
    timing_b64 = _fig_to_base64(fig_timing)
    plt.close(fig_timing)

    # Waveform
    fig_wave = plt.figure(figsize=(14, 3), facecolor=DARK_BG)
    ax = fig_wave.add_subplot(111)
    _draw_waveform(ax, y_vox, sr, beats, speaker_onsets, resultados)
    wave_b64 = _fig_to_base64(fig_wave)
    plt.close(fig_wave)

    # --- Construir tablas de métricas ---
    speaker_rows = ""
    for idx, (spk, data) in enumerate(resultados.items()):
        bio = data["bio"]
        color = _get_speaker_color(spk, idx)
        speaker_rows += f"""
        <tr>
            <td><span class="dot" style="background:{color}"></span>{spk}</td>
            <td class="num"><b>{bio['indice_tecnica']:.1f}</b></td>
            <td class="num"><b>{bio['indice_groove']:.1f}</b></td>
            <td class="num">{data['msp']}</td>
            <td class="num">{bio['accuracy_pct']:.1f}%</td>
            <td class="num">{data['tot']}</td>
            <td>{bio['swing_label']}</td>
            <td class="num">{bio['pct_sincopa']:.1f}%</td>
            <td>{bio['pitch_label']}</td>
            <td class="num">{bio['pct_sustain']:.1f}%</td>
            <td>{bio['dinamica_label']}</td>
        </tr>"""

    # --- Transcripción ---
    transcript_html = ""
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if not text:
            continue
        t_start = seg.get("start", 0)
        mins = int(t_start // 60)
        secs = t_start % 60

        # Color del speaker
        spk_color = TEXT_DIM
        for idx, (s, _) in enumerate(resultados.items()):
            if s == spk:
                spk_color = _get_speaker_color(spk, idx)
                break

        transcript_html += f"""
        <div class="seg">
            <span class="ts">[{mins:02d}:{secs:04.1f}]</span>
            <span class="spk" style="color:{spk_color}">{spk}</span>
            <span class="txt">{text}</span>
        </div>"""

    # --- Veredicto ---
    spks = list(resultados.keys())
    veredicto = ""
    if len(spks) >= 2:
        best_tec = max(spks, key=lambda s: resultados[s]["bio"]["indice_tecnica"])
        best_grv = max(spks, key=lambda s: resultados[s]["bio"]["indice_groove"])
        t_score = resultados[best_tec]["bio"]["indice_tecnica"]
        g_score = resultados[best_grv]["bio"]["indice_groove"]

        if best_tec == best_grv:
            veredicto = f"""<div class="verdict win">
                {best_tec} domina ambos ejes
                (Técnica: {t_score:.1f} | Groove: {g_score:.1f})
            </div>"""
        else:
            veredicto = f"""<div class="verdict split">
                Técnica: {best_tec} ({t_score:.1f}) |
                Groove: {best_grv} ({g_score:.1f}) —
                Estilos distintos
            </div>"""

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FlowMetrics — {input_name}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        background: {DARK_BG};
        color: {TEXT_COLOR};
        font-family: 'Segoe UI', -apple-system, sans-serif;
        line-height: 1.6;
        padding: 2rem;
    }}
    h1 {{
        font-size: 2rem;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #58a6ff, #3fb950);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .subtitle {{ color: {TEXT_DIM}; margin-bottom: 2rem; font-size: 0.9rem; }}
    .meta {{ display: flex; gap: 2rem; flex-wrap: wrap; margin-bottom: 2rem; }}
    .meta-item {{
        background: {PANEL_BG};
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        border: 1px solid {GRID_COLOR};
    }}
    .meta-label {{ color: {TEXT_DIM}; font-size: 0.75rem; text-transform: uppercase; }}
    .meta-value {{ font-size: 1.3rem; font-weight: bold; }}
    .section {{ margin-bottom: 2.5rem; }}
    .section h2 {{
        font-size: 1.3rem;
        color: {ACCENT};
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid {GRID_COLOR};
    }}
    .charts {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }}
    .chart-full {{ grid-column: 1 / -1; }}
    .chart-card {{
        background: {PANEL_BG};
        border-radius: 12px;
        border: 1px solid {GRID_COLOR};
        overflow: hidden;
    }}
    .chart-card img {{ width: 100%; display: block; }}
    table {{
        width: 100%;
        border-collapse: collapse;
        background: {PANEL_BG};
        border-radius: 8px;
        overflow: hidden;
    }}
    th {{
        background: {GRID_COLOR};
        padding: 0.6rem 0.8rem;
        text-align: left;
        font-size: 0.8rem;
        text-transform: uppercase;
        color: {TEXT_DIM};
    }}
    td {{
        padding: 0.5rem 0.8rem;
        border-bottom: 1px solid {GRID_COLOR};
        font-size: 0.9rem;
    }}
    .num {{ font-family: 'Cascadia Code', 'Fira Code', monospace; text-align: right; }}
    .dot {{
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }}
    .verdict {{
        padding: 1rem 1.5rem;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }}
    .verdict.win {{ background: #0d2818; border: 1px solid #3fb950; color: #3fb950; }}
    .verdict.split {{ background: #1c1917; border: 1px solid #f0883e; color: #f0883e; }}
    .transcript {{
        background: {PANEL_BG};
        border-radius: 8px;
        padding: 1rem;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid {GRID_COLOR};
    }}
    .seg {{ padding: 0.2rem 0; font-size: 0.85rem; }}
    .ts {{ color: {TEXT_DIM}; font-family: monospace; margin-right: 0.5rem; }}
    .spk {{ font-weight: bold; margin-right: 0.5rem; }}
    .txt {{ color: {TEXT_COLOR}; }}
    footer {{
        margin-top: 3rem;
        text-align: center;
        color: {TEXT_DIM};
        font-size: 0.75rem;
    }}
</style>
</head>
<body>

<h1>FlowMetrics v1.0</h1>
<p class="subtitle">{input_name} — {method} — {now}</p>

<div class="meta">
    <div class="meta-item">
        <div class="meta-label">Tempo</div>
        <div class="meta-value">{tempo:.0f} BPM</div>
    </div>
    <div class="meta-item">
        <div class="meta-label">Duración</div>
        <div class="meta-value">{len(y_vox)/sr:.0f}s</div>
    </div>
    <div class="meta-item">
        <div class="meta-label">Speakers</div>
        <div class="meta-value">{len(resultados)}</div>
    </div>
    <div class="meta-item">
        <div class="meta-label">Motor</div>
        <div class="meta-value">{method.split('+')[0].strip()}</div>
    </div>
</div>

{veredicto}

<div class="section">
    <h2>Dashboard</h2>
    <div class="charts">
        <div class="chart-card"><img src="data:image/png;base64,{radar_b64}" alt="Radar"></div>
        <div class="chart-card"><img src="data:image/png;base64,{score_b64}" alt="Scoreboard"></div>
        <div class="chart-card"><img src="data:image/png;base64,{sps_b64}" alt="SPS Timeline"></div>
        <div class="chart-card"><img src="data:image/png;base64,{timing_b64}" alt="Micro-timing"></div>
        <div class="chart-card chart-full"><img src="data:image/png;base64,{wave_b64}" alt="Waveform"></div>
    </div>
</div>

<div class="section">
    <h2>Métricas Detalladas</h2>
    <table>
        <thead>
            <tr>
                <th>Speaker</th>
                <th>Técnica</th>
                <th>Groove</th>
                <th>SPS Max</th>
                <th>Precisión</th>
                <th>Sílabas</th>
                <th>Swing</th>
                <th>Síncopa</th>
                <th>Melodía</th>
                <th>Sustain</th>
                <th>Dinámica</th>
            </tr>
        </thead>
        <tbody>
            {speaker_rows}
        </tbody>
    </table>
</div>

<div class="section">
    <h2>Transcripción</h2>
    <div class="transcript">
        {transcript_html}
    </div>
</div>

<footer>
    Generado por FlowMetrics v1.0 — Telemetría acústica para freestyle rap
</footer>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  ✅ Reporte HTML: {output_path}")
    return output_path
