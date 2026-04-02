"""
Módulo de output: TXT y JSON estructurado.

Formatos de exportación para los resultados del análisis.
"""

import json
import os

import numpy as np

from . import __version__


class _NumpyEncoder(json.JSONEncoder):
    """Encoder que convierte tipos numpy a tipos nativos de Python."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_transcript_txt(segments, input_name, method, output_path):
    """Guarda transcripción en formato texto plano."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"FlowMetrics v{__version__} — {method}\n")
        f.write(f"Source: {input_name}\n\n")
        for s in segments:
            spk = s.get("speaker", "UNKNOWN")
            text = s.get("text", "").strip()
            if not text:
                continue
            mins = int(s['start'] // 60)
            secs = s['start'] % 60
            f.write(f"[{spk}] [{mins:02d}:{secs:04.1f}] {text}\n")
    print(f"  📄 Transcripción: {output_path}")


def save_json(resultados, segments, tempo, input_name, method,
              output_path):
    """Exporta resultados completos en JSON estructurado.

    Incluye métricas por speaker, transcripción con timestamps,
    y metadata del análisis. Útil para integración con otros sistemas.
    """
    # Limpiar datos no serializables (numpy arrays) de los resultados
    clean_results = {}
    for spk, data in resultados.items():
        bio = {}
        for k, v in data["bio"].items():
            # Excluir arrays numpy grandes (contours para visualización)
            if isinstance(v, np.ndarray):
                continue
            # Convertir escalares numpy a Python nativos
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            bio[k] = v
        clean_results[spk] = {
            "bio": bio,
            "sps_max": data["msp"],
            "accuracy_pct": data["acc"],
            "total_silabas": data["tot"],
            "peak_time": data["pt"],
        }

    # Transcripción limpia
    clean_segments = []
    for s in segments:
        text = s.get("text", "").strip()
        if not text:
            continue
        clean_segments.append({
            "speaker": s.get("speaker", "UNKNOWN"),
            "start": round(s.get("start", 0), 2),
            "end": round(s.get("end", 0), 2),
            "text": text,
        })

    output = {
        "version": __version__,
        "source": input_name,
        "method": method,
        "tempo_bpm": round(tempo, 1),
        "speakers": clean_results,
        "transcript": clean_segments,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    print(f"  📊 JSON: {output_path}")
