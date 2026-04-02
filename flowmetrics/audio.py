"""
Módulo de carga y separación de audio.

Responsabilidades:
- Buscar archivos de audio en pendings/ o en la raíz del proyecto
- Separar vocals/instrumental con Demucs (con cache inteligente)
- Cargar audio normalizado para análisis
"""

import os
import sys
import librosa
import numpy as np

from .config import (
    SCRIPT_DIR, INPUT_EXTENSIONS, PENDINGS_DIR,
    DEMUCS_MODEL, DEMUCS_OUTPUT_DIR, SR,
)
from .patches import patch_torchaudio


def find_input_file(path=None):
    """Busca archivo de audio a procesar.

    Args:
        path: ruta explícita (desde CLI). Si se da, se usa directamente.

    Prioridad cuando path=None:
        1. Cualquier archivo de audio en pendings/
        2. Archivo 'batalla.*' en el directorio raíz
    """
    if path:
        if os.path.isfile(path):
            return os.path.abspath(path)
        raise FileNotFoundError(f"No se encontró: {path}")

    if os.path.isdir(PENDINGS_DIR):
        for fname in os.listdir(PENDINGS_DIR):
            if any(fname.lower().endswith(ext) for ext in INPUT_EXTENSIONS):
                return os.path.join(PENDINGS_DIR, fname)

    for ext in INPUT_EXTENSIONS:
        f = os.path.join(SCRIPT_DIR, "batalla" + ext)
        if os.path.isfile(f):
            return f

    return None


def separate_audio(input_path):
    """Separa audio en vocals + instrumental usando Demucs.

    Usa cache: si los stems ya existen y son más nuevos que el fuente,
    los reutiliza sin re-procesar.

    Returns:
        (instrumental_path, vocals_path)
    """
    print("\n🧠 [PASO 0] Demucs (Separación)...")
    patch_torchaudio()

    fname = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(DEMUCS_OUTPUT_DIR, DEMUCS_MODEL, fname)
    voc = os.path.join(out_dir, "vocals.wav")
    inst = os.path.join(out_dir, "no_vocals.wav")

    if os.path.exists(voc) and os.path.exists(inst):
        source_mtime = os.path.getmtime(input_path)
        cache_mtime = min(os.path.getmtime(voc), os.path.getmtime(inst))
        if cache_mtime > source_mtime:
            print("  ✅ Cache válido. Usando stems existentes.")
            return inst, voc
        else:
            print("  🔄 Fuente más reciente que cache. Re-procesando...")

    print("  ⏳ Separando (esto puede tardar varios minutos)...")
    try:
        from demucs.separate import main as demucs_main
        sys.argv = [
            "demucs", "-n", DEMUCS_MODEL,
            "--two-stems=vocals", "-o", DEMUCS_OUTPUT_DIR, input_path,
        ]
        demucs_main()
        return inst, voc
    except Exception as e:
        print(f"  ❌ Error Demucs: {e}")
        sys.exit(1)


def load_audio(path, sr=SR):
    """Carga audio mono normalizado.

    Returns:
        (y, sr) — señal numpy y sample rate
    """
    y, sr_out = librosa.load(path, sr=sr, mono=True)
    return y, sr_out
