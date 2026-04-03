"""
Configuración global de FlowMetrics.

Centraliza constantes, rutas y umbrales para que todos los módulos
lean de un único lugar. Los valores se pueden sobreescribir desde CLI.
"""

import os
import sys
import warnings

# Suprimir warnings ruidosos ANTES de importar pyannote/torch
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*libtorchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

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
import logging
for _logger_name in ['nemo_logger', 'nemo', 'pytorch_lightning', 'lightning.fabric', 'lightning.pytorch']:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

# Windows: forzar UTF-8 para emojis en consola
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ============================================================================
# CARGA DE .env (tokens y secretos)
# ============================================================================
# Lee el archivo .env del directorio del proyecto para cargar variables
# de entorno como HF_TOKEN sin exponerlas en el código fuente.

_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_env_path = os.path.join(_SCRIPT_DIR, ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# ============================================================================
# RUTAS
# ============================================================================

SCRIPT_DIR = _SCRIPT_DIR
INPUT_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg"]
PENDINGS_DIR = os.path.join(SCRIPT_DIR, "pendings")
DEMUCS_MODEL = "htdemucs"
DEMUCS_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "separated")
OUTPUT_DIR = SCRIPT_DIR
VOICEDB_PATH = os.path.join(SCRIPT_DIR, "voicedb.json")

# ============================================================================
# AUDIO
# ============================================================================

SR = 22050
BEAT_MATCH_THRESHOLD_S = 0.05

# ============================================================================
# TRANSCRIPCIÓN
# ============================================================================

WHISPERX_MODEL = "large-v2"
WHISPER_FALLBACK_MODEL = "small"
COMPUTE_TYPE = "int8"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ============================================================================
# MODELOS NVIDIA (requieren NeMo, funciona en WSL2)
# ============================================================================

SORTFORMER_MODEL_NAME = "nvidia/diar_streaming_sortformer_4spk-v2"
TITANET_MODEL_NAME = "nvidia/speakerverification_en_titanet_large"

# ============================================================================
# UMBRALES DE SIMILITUD COSENO POR TIPO DE EMBEDDING
# Cada modelo genera embeddings en un espacio distinto con distinta
# capacidad discriminativa, por eso necesitan umbrales distintos.
# ============================================================================

TITANET_MATCH_THRESHOLD = 0.7     # TitaNet 192 dims — muy discriminativo
EMBEDDING_MATCH_THRESHOLD = 0.75  # pyannote 256 dims — alta discriminación
MFCC_MATCH_THRESHOLD = 0.65      # MFCC 40 dims — menos discriminativo

# ============================================================================
# VISUALIZACIÓN
# ============================================================================

SPEAKER_COLORS = {
    "SPEAKER_00": "#3fb950",
    "SPEAKER_01": "#58a6ff",
    "SPEAKER_02": "#d2a8ff",
    "SPEAKER_03": "#f0883e",
    "UNKNOWN":    "#8b949e",
}

EXTRA_COLORS = ["#3fb950", "#58a6ff", "#d2a8ff", "#f0883e", "#ff7b72",
                "#f778ba", "#56d4dd", "#dbab09"]

# ============================================================================
# DETECCIÓN DE BACKENDS DISPONIBLES
# ============================================================================
# NeMo imprime mensajes ruidosos durante el import (Megatron, OneLogger,
# torch.distributed) que no se pueden suprimir con logging/warnings porque
# usan print() directo. Los capturamos redirigiendo stderr/stdout.

BACKEND_SORTFORMER = False
BACKEND_PYANNOTE = False

try:
    import io as _io
    _old_stdout, _old_stderr = sys.stdout, sys.stderr
    sys.stdout = _io.StringIO()
    sys.stderr = _io.StringIO()
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel, EncDecSpeakerLabelModel
        BACKEND_SORTFORMER = True
    except ImportError:
        pass
    finally:
        sys.stdout = _old_stdout
        sys.stderr = _old_stderr
except Exception:
    pass

try:
    from whisperx.diarize import DiarizationPipeline as _DiarPipeline
    BACKEND_PYANNOTE = True
except ImportError:
    pass
