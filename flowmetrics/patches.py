"""
Parches de compatibilidad para torchaudio en Windows/Python 3.13.

Demucs depende de torchaudio, que en Windows puede fallar por falta
de backend de audio. Este módulo reemplaza torchaudio.load/save con
soundfile como backend alternativo.
"""

import sys


def patch_torchaudio():
    """Aplica monkey-patch a torchaudio usando soundfile como backend."""
    try:
        import torchaudio as ta
        import torch
        import soundfile as sf

        if hasattr(ta, '_patched_by_flowmetrics'):
            return

        try:
            from torchcodec.decoders import AudioDecoder
            return
        except Exception:
            pass

        def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                     channels_first=True, format=None, buffer_size=4096, backend=None):
            data, sr = sf.read(str(uri), dtype='float32', start=frame_offset,
                               stop=frame_offset + num_frames if num_frames > 0 else None,
                               always_2d=True)
            tensor = torch.from_numpy(data.T)
            if not channels_first:
                tensor = tensor.T
            return tensor, sr

        def _sf_save(uri, src, sample_rate, channels_first=True, **kwargs):
            import numpy as np
            if isinstance(src, torch.Tensor):
                data = src.detach().cpu().numpy()
            else:
                data = np.asarray(src)
            if channels_first and data.ndim == 2:
                data = data.T
            sf.write(str(uri), data, sample_rate)

        ta.load = _sf_load
        ta.save = _sf_save
        ta._patched_by_flowmetrics = True

    except Exception:
        import types
        dummy = types.ModuleType("torchaudio")
        sys.modules["torchaudio"] = dummy
        dummy.load = lambda *a, **k: None
        dummy.save = lambda *a, **k: None
        try:
            import soundfile as sf
            import torch

            def _mock_load(uri, *a, **k):
                d, s = sf.read(str(uri), dtype='float32', always_2d=True)
                return torch.from_numpy(d.T), s
            dummy.load = _mock_load
        except Exception:
            pass
