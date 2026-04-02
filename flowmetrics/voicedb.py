"""
Base de datos de huellas vocales e identificación de MCs.

Gestiona el almacenamiento y comparación de embeddings de voz para
reconocer MCs entre batallas. Soporta embeddings de TitaNet (192d),
pyannote (256d) y MFCC (40d).
"""

import os
import json
import numpy as np

from .config import (
    VOICEDB_PATH, TITANET_MATCH_THRESHOLD,
    EMBEDDING_MATCH_THRESHOLD, MFCC_MATCH_THRESHOLD,
)


def load_voicedb(path=None):
    """Carga la base de datos de huellas vocales.

    Returns:
        dict {nombre_mc: {"embedding": [...], "type": str, "battles": int}}
    """
    db_path = path or VOICEDB_PATH
    if os.path.exists(db_path):
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  ⚠️  Error leyendo voicedb: {e}")
    return {}


def save_voicedb(db, path=None):
    """Persiste la base de datos de huellas vocales."""
    db_path = path or VOICEDB_PATH
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def cosine_similarity(a, b):
    """Similitud coseno entre dos vectores de embedding.

    Se usa en vez de distancia euclidiana porque los embeddings de
    speaker viven en un espacio donde la dirección importa más que
    la magnitud.
    """
    a, b = np.array(a), np.array(b)
    if a.shape != b.shape:
        return 0.0
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def identify_speakers(embeddings, voicedb, emb_type="pyannote"):
    """Mapea SPEAKER_XX → nombre real comparando embeddings con la DB.

    Para cada speaker con embedding, calcula similitud coseno contra
    todos los MCs registrados del mismo tipo. Si supera el umbral,
    asigna el nombre conocido.

    Returns:
        dict {speaker_label: nombre_real_o_"SPEAKER_XX (Nuevo)"}
    """
    threshold_map = {
        "titanet": TITANET_MATCH_THRESHOLD,
        "pyannote": EMBEDDING_MATCH_THRESHOLD,
        "mfcc": MFCC_MATCH_THRESHOLD,
    }
    threshold = threshold_map.get(emb_type, EMBEDDING_MATCH_THRESHOLD)
    speaker_map = {}
    used_names = set()

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
                print(f"  ❓ {spk_label} no coincide (mejor: {best_name} @ {best_sim:.3f})")
            else:
                print(f"  ❓ {spk_label} — no hay MCs compatibles (tipo={emb_type})")

    return speaker_map


def register_new_speakers(embeddings, speaker_map, voicedb, emb_type="pyannote"):
    """Pregunta al usuario si quiere registrar speakers no identificados.

    Para cada speaker "(Nuevo)", ofrece al usuario darle nombre. Si el MC
    ya existía, promedia embeddings (media móvil) para robustecer el
    reconocimiento futuro.

    Returns:
        voicedb actualizada (también guardada a disco)
    """
    nuevos = {label: name for label, name in speaker_map.items()
              if name.endswith("(Nuevo)")}

    if not nuevos:
        return voicedb

    print("\n" + "=" * 60)
    print("  📋 REGISTRO DE MCs")
    print("=" * 60)
    print("  Speakers no identificados. ¿Registrar? (vacío = saltar)")
    print("  Tip: Etiquetar host/presentador como 'Host'\n")

    for spk_label in nuevos:
        if spk_label not in embeddings:
            continue
        nombre = input(f"  Nombre para {spk_label}: ").strip()
        if not nombre:
            continue

        emb = embeddings[spk_label]

        # Match case-insensitive para evitar duplicados
        nombre_real = nombre
        for existing_name in voicedb:
            if existing_name.lower() == nombre.lower():
                nombre_real = existing_name
                break
        nombre = nombre_real

        if nombre in voicedb and voicedb[nombre].get("type") == emb_type:
            # Media móvil del embedding
            old_emb = np.array(voicedb[nombre]["embedding"])
            new_emb = np.array(emb)
            n = voicedb[nombre]["battles"]
            averaged = ((old_emb * n) + new_emb) / (n + 1)
            voicedb[nombre]["embedding"] = averaged.tolist()
            voicedb[nombre]["battles"] = n + 1
            print(f"  ✅ {nombre} actualizado (batalla #{n + 1})")
        else:
            if nombre in voicedb:
                print(f"  ⚠️  {nombre} tenía tipo '{voicedb[nombre].get('type', '?')}', "
                      f"reemplazando con '{emb_type}'")
            voicedb[nombre] = {
                "embedding": list(emb) if not isinstance(emb, list) else emb,
                "type": emb_type,
                "battles": 1,
            }
            print(f"  ✅ {nombre} registrado ({emb_type}, {len(emb)} dims)")

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
