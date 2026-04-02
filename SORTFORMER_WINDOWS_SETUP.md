# Sortformer v2 en Windows (CPU) — Guia de instalacion

## Contexto

NVIDIA NeMo **no soporta Windows oficialmente**. Sin embargo, Sortformer v2
(diarizacion end-to-end, DER ~6.57%) y TitaNet (embeddings de speaker, 192 dims)
**si funcionan en Windows con CPU** aplicando los workarounds de esta guia.

Probado con:
- Windows 11 Home 10.0.22631
- Python 3.13.12 (Microsoft Store)
- PyTorch 2.8.0+cpu
- NeMo 2.7.2
- Abril 2026

## Rendimiento esperado

- **CPU**: ~8-10 segundos para diarizar 2 minutos de audio (vs ~1s en GPU)
- **Sin GPU**: Funciona perfectamente, solo mas lento
- **Modelos**: Se descargan una sola vez (~200MB Sortformer + ~100MB TitaNet)
  y quedan cacheados en `~/.cache/huggingface/hub/`

## Pasos de instalacion

### 1. Prerequisitos

Tener un venv con PyTorch (CPU) y las dependencias base del proyecto:

```bash
python -m venv .venv
.venv\Scripts\activate

# PyTorch CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Instalar NeMo base

```bash
pip install nemo_toolkit
```

### 3. Workaround para `editdistance` (CRITICO)

El paquete `editdistance` no tiene wheels para Python 3.13 en Windows y falla
al compilar el codigo C. La solucion es usar `editdistance-s` (reimplementacion
compatible) y crear un shim:

```bash
# Instalar el reemplazo
pip install editdistance-s

# Crear shim para que NeMo encuentre el modulo
python -c "import sysconfig; open(sysconfig.get_paths()['purelib']+'/editdistance.py','w').write('from editdistance_s import distance as eval\n')"
```

### 4. Instalar dependencias ASR de NeMo

NeMo declara muchas dependencias opcionales que no se instalan con el paquete base.
Para ASR/diarizacion se necesitan:

```bash
# Framework core
pip install "hydra-core>1.3,<=1.3.2" "omegaconf<=2.3" fiddle

# Lightning (IMPORTANTE: version <= 2.4.0 para compatibilidad con nv_one_logger)
pip install "lightning>=2.2.1,<=2.4.0"

# NeMo logger (requerido desde NeMo 2.7+)
pip install "nv_one_logger_core>=2.3.1"
pip install "nv_one_logger_training_telemetry>=2.3.1"
pip install "nv_one_logger_pytorch_lightning_integration>=2.3.1"

# ASR deps
pip install "lhotse>=1.32.2" "sentencepiece<1.0.0" "jiwer>=3.1.0,<4.0.0"
pip install braceexpand kaldi-python-io "kaldialign<=0.9.1"
pip install "webdataset>=0.2.86" datasets ipython
```

### 5. Verificar instalacion

```python
import warnings
warnings.filterwarnings('ignore')
from nemo.collections.asr.models import SortformerEncLabelModel, EncDecSpeakerLabelModel

# Cargar Sortformer (diarizacion)
model = SortformerEncLabelModel.from_pretrained(
    'nvidia/diar_streaming_sortformer_4spk-v2',
    map_location='cpu'
)
print(f"Sortformer OK - device: {next(model.parameters()).device}")

# Cargar TitaNet (embeddings de speaker)
titanet = EncDecSpeakerLabelModel.from_pretrained(
    'nvidia/speakerverification_en_titanet_large',
    map_location='cpu'
)
print(f"TitaNet OK - device: {next(titanet.parameters()).device}")
```

## Script de instalacion rapida (one-liner)

Para instalar todo de una vez en un venv que ya tiene PyTorch:

```bash
pip install nemo_toolkit editdistance-s "hydra-core>1.3,<=1.3.2" "omegaconf<=2.3" fiddle "lightning>=2.2.1,<=2.4.0" "nv_one_logger_core>=2.3.1" "nv_one_logger_training_telemetry>=2.3.1" "nv_one_logger_pytorch_lightning_integration>=2.3.1" "lhotse>=1.32.2" "sentencepiece<1.0.0" "jiwer>=3.1.0,<4.0.0" braceexpand kaldi-python-io "kaldialign<=0.9.1" "webdataset>=0.2.86" datasets ipython && python -c "import sysconfig; open(sysconfig.get_paths()['purelib']+'/editdistance.py','w').write('from editdistance_s import distance as eval\n')"
```

## Problemas conocidos

### Warnings inofensivos (se suprimen automaticamente en FlowMetrics)

- `Megatron num_microbatches_calculator not found` — NeMo busca Megatron-LM
  para entrenamiento distribuido. No se necesita para inferencia.
- `Redirects are currently not supported in Windows` — torch.distributed no
  soporta redireccion de stdout en Windows. No afecta.
- `setup_training_data() / setup_validation_data()` — el modelo avisa que no
  hay datos de entrenamiento configurados. Normal para inferencia.
- `torchcodec is not installed correctly` — pyannote busca torchcodec para
  decodificacion de audio. No se necesita porque usamos librosa/soundfile.

### Lightning > 2.4.0 rompe nv_one_logger

Si `lightning` se actualiza a > 2.4.0, el import de NeMo falla con:
```
TypeError: `OneLoggerPTLTrainer.save_checkpoint: weights_only must be a supertype of...`
```
Solucion: `pip install "lightning<=2.4.0"`

### editdistance no compila en Python 3.13

El paquete original `editdistance` requiere compilar C y falla en Python 3.13
Windows. La solucion es `editdistance-s` + shim (ver paso 3).

### Sin GPU CUDA

Sortformer funciona en CPU. Es ~10x mas lento que GPU pero totalmente funcional.
Para audios de 2-5 minutos (tipico de batallas), tarda ~10-30 segundos.

## Modelos utilizados

| Modelo | Tarea | Dims | HuggingFace |
|--------|-------|------|-------------|
| Sortformer v2 | Diarizacion (quien habla cuando) | - | `nvidia/diar_streaming_sortformer_4spk-v2` |
| TitaNet Large | Embeddings de speaker (identidad vocal) | 192 | `nvidia/speakerverification_en_titanet_large` |
