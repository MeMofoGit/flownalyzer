"""
Entry point: python -m flowmetrics

Orquesta todo el pipeline:
1. Buscar/separar audio
2. Análisis instrumental (beats, grid)
3. Detección de sílabas (onsets)
4. Transcripción + diarización
5. Identificación de MCs (VoiceDB)
6. Biometría de flow por speaker
7. Visualización + output
"""

import os
import sys
import warnings
import numpy as np
import librosa

# Suprimir warnings ruidosos de torchcodec (no afectan funcionalidad)
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*libtorchcodec.*")

# Suprimir logs [NeMo W/I ...] y mensajes verbose de PyTorch Lightning
import logging as _logging
for _logger_name in ['nemo_logger', 'nemo', 'pytorch_lightning', 'lightning.fabric', 'lightning.pytorch']:
    _logging.getLogger(_logger_name).setLevel(_logging.ERROR)

# Windows usa cp1252 por defecto, lo que rompe emojis en la consola.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from .cli import parse_args
from .config import SCRIPT_DIR, OUTPUT_DIR, BACKEND_SORTFORMER, BACKEND_PYANNOTE
from .patches import patch_torchaudio
from .audio import find_input_file, separate_audio, load_audio
from .analysis import (
    detect_beats, generate_grid, detect_onsets,
    calc_metrics, extract_speaker_audio, analizar_biometria_flow,
    map_onsets_to_speakers, filter_minor_speakers,
)
from .transcription import (
    transcribe_whisperx, transcribe_whisper_fallback, spectral_diarize,
)
from .voicedb import (
    load_voicedb, identify_speakers, register_new_speakers,
    rename_segments, save_voicedb, manage_voicedb_interactive,
)
from .output import save_transcript_txt, save_json
from .visualization import generate_dashboard, generate_html_report
from . import __version__


def _bar(value, max_val=100, width=20):
    filled = int((value / max_val) * width)
    return f"[{'|' * filled}{' ' * (width - filled)}]"


def _print_scoreboard(resultados, segments):
    """Imprime scoreboard dual en consola."""
    print("\n" + "=" * 62)
    print("  🏆 SCOREBOARD — TÉCNICA vs GROOVE")
    print("=" * 62)

    for spk, data in resultados.items():
        bio = data["bio"]
        t = bio["indice_tecnica"]
        g = bio["indice_groove"]

        # Buscar línea del peak SPS
        rhyme = ""
        for s in segments:
            if s.get("speaker") == spk and s['start'] <= data["pt"] <= s['end'] + 0.5:
                rhyme = s['text'].strip()
                break

        print(f"\n  {'=' * 58}")
        print(f"  {spk}")
        print(f"  {'=' * 58}")
        print(f"  TECNICA  {_bar(t)}  {t:.1f}/100")
        print(f"  GROOVE   {_bar(g)}  {g:.1f}/100")
        print(f"  {'-' * 58}")
        print(f"  Tecnica: precision={bio['pts_precision']:.0f}"
              f" + velocidad={bio['pts_velocidad']:.0f}"
              f" + volumen={bio['pts_volumen']:.0f}")
        print(f"    SPS Max: {data['msp']:<2}  |  Acc: {data['acc']:.1f}%"
              f"  |  Silabas: {data['tot']}")
        print(f"  {'-' * 58}")
        print(f"  Groove:  swing={bio['pts_swing']:.0f}"
              f" + sinc={bio['pts_sincopa']:.0f}"
              f" + smooth={bio['pts_smooth']:.0f}"
              f" + sust={bio['pts_sustain']:.0f}"
              f" + din={bio['pts_dinamica']:.0f}")
        print(f"    Swing:    {bio['swing_label']:<13} "
              f"(μ={bio['delta_media_ms']:.1f}ms, σ={bio['delta_std_ms']:.1f}ms)")
        print(f"    Sincopa:  {bio['sincopa_label']:<13} "
              f"({bio['pct_sincopa']:.1f}% en contratiempos)")
        print(f"    Melodia:  {bio['pitch_label']:<13} "
              f"(smooth={bio['smooth_score']:.0f}, voiced={bio['pct_voiced']:.0f}%, "
              f"dF0={bio['median_delta_f0']:.1f}Hz)")
        print(f"    Sustain:  {bio['chicleadas']} chicleo(s)      "
              f"({bio['pct_sustain']:.1f}% del tiempo)")
        print(f"    Dinamica: {bio['dinamica_label']:<13} "
              f"(var={bio['varianza_rms']:.4f})")
        if rhyme:
            print(f"    Peak:     \"{rhyme[:55]}\"")

    # Veredicto
    if len(resultados) >= 2:
        spks = list(resultados.keys())
        print(f"\n  {'=' * 58}")
        print(f"  VEREDICTO")
        print(f"  {'=' * 58}")

        best_tec = max(spks, key=lambda s: resultados[s]["bio"]["indice_tecnica"])
        best_grv = max(spks, key=lambda s: resultados[s]["bio"]["indice_groove"])
        t_score = resultados[best_tec]["bio"]["indice_tecnica"]
        g_score = resultados[best_grv]["bio"]["indice_groove"]

        print(f"  Tecnica: {best_tec} ({t_score:.1f})")
        print(f"  Groove:  {best_grv} ({g_score:.1f})")

        if best_tec == best_grv:
            print(f"\n  {best_tec} domina ambos ejes.")
        else:
            print(f"\n  Estilos distintos — depende del criterio del jurado.")
        print()


def main(argv=None):
    args = parse_args(argv)

    # Modo gestión de VoiceDB (no requiere audio)
    if args.manage_voicedb:
        manage_voicedb_interactive()
        return

    patch_torchaudio()

    print("=" * 62)
    print(f"  🎤 FlowMetrics v{__version__}")
    print("=" * 62)

    # Backends disponibles
    print(f"  [Backend] Sortformer: {'✅' if BACKEND_SORTFORMER else '—'}")
    print(f"  [Backend] pyannote:   {'✅' if BACKEND_PYANNOTE else '—'}")
    if not BACKEND_SORTFORMER and not BACKEND_PYANNOTE:
        print(f"  [Backend] Fallback espectral (MFCC + KMeans)")

    # ── 1. Buscar y separar audio ──
    infile = find_input_file(args.audio)
    if not infile:
        sys.exit("❌ No se encontró archivo de audio. "
                 "Usa: python -m flowmetrics <archivo.wav>")

    print(f"\n  📦 Archivo: {os.path.basename(infile)}")
    inst_path, vox_path = separate_audio(infile)

    # ── 2. Análisis instrumental ──
    y_inst, sr = load_audio(inst_path)
    tempo, beats = detect_beats(y_inst, sr)
    duration = librosa.get_duration(y=y_inst, sr=sr)
    grid = generate_grid(tempo, beats, duration)
    print(f"  🎵 Tempo: {tempo:.0f} BPM | Beats: {len(beats)} | Grid: {len(grid)}")

    # ── 3. Detección de sílabas ──
    y_vox, sr_vox = load_audio(vox_path)
    all_onsets = detect_onsets(y_vox, sr_vox)
    print(f"  🗣️  Sílabas detectadas: {len(all_onsets)}")

    # ── 4. VoiceDB ──
    voicedb = load_voicedb()
    if voicedb:
        print(f"\n  📂 VoiceDB: {len(voicedb)} MC(s) — {', '.join(voicedb.keys())}")
    else:
        print(f"\n  📂 VoiceDB vacía")

    # ── 5. Transcripción + Diarización ──
    segments, embeddings, emb_type = transcribe_whisperx(
        vox_path,
        min_speakers=args.speakers,
        max_speakers=args.max_speakers,
        vad_sensitivity=args.vad_sensitivity,
    )

    if segments is None:
        segments, embeddings, emb_type = transcribe_whisper_fallback(vox_path)
    elif emb_type == "whisperx_no_diarize":
        print("  🔬 Aplicando diarización espectral...")
        segments, mfcc_embeddings = spectral_diarize(
            vox_path, segments, num_speakers=args.speakers,
        )
        embeddings = mfcc_embeddings
        emb_type = "mfcc"

    method_names = {
        "titanet": "WhisperX + Sortformer v2 + TitaNet",
        "pyannote": "WhisperX + pyannote",
        "mfcc": "Whisper + Spectral (MFCC+KMeans)",
        "failed": "Whisper (Sin Diarización)",
    }
    method = method_names.get(emb_type, f"WhisperX ({emb_type})")

    # ── 6. Identificar speakers ──
    speaker_map = {}
    if embeddings:
        print(f"\n  🔍 Identificando speakers ({emb_type})...")
        speaker_map = identify_speakers(embeddings, voicedb, emb_type=emb_type)
        segments = rename_segments(segments, speaker_map)

    # ── 6b. Dedup final — eliminar segmentos duplicados del chunking ──
    # Con chunks cortos (30s), la misma frase puede aparecer en dos chunks
    # distintos con timestamps ligeramente diferentes tras la alineación.
    if segments:
        import re as _re
        _norm = lambda t: _re.sub(r'[^\w\s]', '', t.strip().lower())
        seen = []  # (start, normalized_text)
        clean_segments = []
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            norm_text = _norm(text)
            start = seg.get("start", 0)
            is_dup = False
            for prev_start, prev_text in seen:
                if abs(start - prev_start) < 5.0 and (
                    prev_text == norm_text or
                    (len(prev_text) > 8 and prev_text in norm_text) or
                    (len(norm_text) > 8 and norm_text in prev_text)
                ):
                    is_dup = True
                    break
            if not is_dup:
                clean_segments.append(seg)
                seen.append((start, norm_text))
        n_removed = len(segments) - len(clean_segments)
        if n_removed > 0:
            print(f"  🧹 Dedup final: {n_removed} segmentos duplicados eliminados")
        segments = clean_segments

    # ── 7. Mapear onsets a speakers ──
    speaker_onsets = map_onsets_to_speakers(all_onsets, segments)
    principales, menores = filter_minor_speakers(speaker_onsets, grid)

    if menores:
        names = ", ".join(f"{s} ({len(o)} sil)" for s, o in menores.items())
        print(f"\n  🔇 Excluidos del scoreboard: {names}")

    # ── 8. Biometría por speaker ──
    resultados = {}
    for spk, onsets in principales.items():
        if not onsets:
            continue
        m = calc_metrics(np.array(onsets), grid)
        spk_audio = extract_speaker_audio(y_vox, sr_vox, segments, spk)
        bio = analizar_biometria_flow(
            spk_audio, np.array(onsets), grid, sr_vox,
            sps_max=m["sps_max"], total_silabas=m["total_onsets"],
        )
        resultados[spk] = {
            "bio": bio, "msp": m["sps_max"],
            "acc": m["accuracy_pct"], "tot": m["total_onsets"],
            "pt": m["peak_time"],
        }

    # ── 9. Scoreboard en consola ──
    _print_scoreboard(resultados, segments)

    # ── 10. Cobertura ──
    if segments:
        audio_dur = librosa.get_duration(y=y_vox, sr=sr_vox)
        last_seg_end = max(s.get("end", 0) for s in segments)
        cobertura = (last_seg_end / audio_dur * 100) if audio_dur > 0 else 0
        if cobertura < 85:
            print(f"\n  ⚠️  Cobertura: {cobertura:.0f}% "
                  f"({last_seg_end:.0f}s de {audio_dur:.0f}s)")

    # ── 11. Registro de MCs ──
    if embeddings and not args.no_register:
        voicedb = register_new_speakers(
            embeddings, speaker_map, voicedb, emb_type=emb_type,
        )
        segments = rename_segments(segments, speaker_map)

    # ── 12. Outputs ──
    out_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    input_name = os.path.basename(infile)
    base_name = os.path.splitext(input_name)[0]

    print(f"\n  📤 Generando outputs...")

    if "txt" in args.format:
        save_transcript_txt(
            segments, input_name, method,
            os.path.join(out_dir, "transcripcion.txt"),
        )

    if "json" in args.format:
        save_json(
            resultados, segments, tempo, input_name, method,
            os.path.join(out_dir, f"{base_name}_metrics.json"),
        )

    if "png" in args.format:
        generate_dashboard(
            y_vox, sr_vox, beats, principales, resultados,
            os.path.join(out_dir, "flow_graph.png"),
        )

    if "html" in args.format:
        generate_html_report(
            resultados, segments, principales,
            y_vox, sr_vox, beats, tempo, input_name, method,
            os.path.join(out_dir, f"{base_name}_report.html"),
        )

    print(f"\n  ✅ Análisis completo.")


if __name__ == "__main__":
    main()
