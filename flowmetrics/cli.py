"""
CLI de FlowMetrics.

Uso:
    python -m flowmetrics batalla.wav
    python -m flowmetrics batalla.wav --speakers 3 --format html json
    python -m flowmetrics --help
"""

import argparse
import os
import sys

from . import __version__


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="flowmetrics",
        description="FlowMetrics v{} — Telemetría acústica para batallas de freestyle rap".format(__version__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Ejemplos:
  python -m flowmetrics batalla.wav
  python -m flowmetrics audio.mp3 --speakers 3
  python -m flowmetrics audio.wav --format html json --output-dir results/
  python -m flowmetrics --no-register   # Skip registro interactivo de MCs
""",
    )

    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Archivo de audio a analizar (wav/mp3/flac/ogg). "
             "Si se omite, busca en pendings/ o batalla.*",
    )

    parser.add_argument(
        "--speakers", "-s",
        type=int,
        default=3,
        help="Número mínimo de speakers esperados (default: 3, "
             "incluye host + 2 MCs). Pyannote puede detectar más.",
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Máximo de speakers (default: auto). Fijar si se detectan "
             "demasiados speakers fantasma.",
    )

    parser.add_argument(
        "--format", "-f",
        nargs="+",
        default=["txt", "png"],
        choices=["txt", "json", "html", "png"],
        help="Formatos de salida (default: txt png). Opciones: txt json html png",
    )

    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Directorio de salida (default: mismo que el proyecto)",
    )

    parser.add_argument(
        "--no-register",
        action="store_true",
        help="No preguntar por registro de MCs nuevos",
    )

    parser.add_argument(
        "--vad-sensitivity",
        type=str,
        default="battle",
        choices=["low", "normal", "battle", "aggressive"],
        help="Sensibilidad del VAD para detección de voz (default: battle). "
             "Usar 'aggressive' si se pierden secciones con ruido de público.",
    )

    parser.add_argument(
        "--manage-voicedb",
        action="store_true",
        help="Gestionar VoiceDB: listar, renombrar o eliminar MCs registrados. "
             "No requiere archivo de audio.",
    )

    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"FlowMetrics v{__version__}",
    )

    return parser.parse_args(argv)
