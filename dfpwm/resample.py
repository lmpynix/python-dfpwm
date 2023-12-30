from typing import BinaryIO, TypeVar

import numpy as np

from .constants import SAMPLE_RATE

T = TypeVar("T", bound=np.dtype)


def resample_from_file(io: "BinaryIO", target_samplerate=SAMPLE_RATE, dtype: T = np.float32) -> \
        tuple["np.ndarray", float]:
    import warnings
    import librosa
    warnings.warn("resample_from_file is deprecated, use dfpwm.resample instead")
    return librosa.load(io, dtype=dtype, sr=target_samplerate)


def resample(data: np.ndarray, origin_samplerate: float, target_sample_rate=SAMPLE_RATE, dtype: T = np.float32) -> \
        np.ndarray[T]:
    import librosa
    return librosa.resample(data, orig_sr=origin_samplerate, target_sr=target_sample_rate).astype(dtype)
