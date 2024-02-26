import typing as tp

import torch

from librosa import filters
import numpy as np
from collections import Counter

def get_fftfreq(
        sr: int = 44100,
        n_fft: int = 2048
) -> torch.Tensor:
    """
    Torch workaround of librosa.fft_frequencies
    """
    out = sr * torch.fft.fftfreq(n_fft)[:n_fft // 2 + 1]
    out[-1] = sr // 2
    return out


def get_subband_indices(
        freqs: torch.Tensor,
        splits: tp.List[tp.Tuple[int, int]],
) -> tp.List[tp.Tuple[int, int]]:
    """
    Computes subband frequency indices with given bandsplits
    """
    indices = []
    start_freq, start_index = 0, 0
    for end_freq, step in splits:
        bands = torch.arange(start_freq + step, end_freq + step, step)
        start_freq = end_freq
        for band in bands:
            end_index = freqs[freqs < band].shape[0]
            indices.append((start_index, end_index))
            start_index = end_index
    indices.append((start_index, freqs.shape[0]))
    return indices


def freq2bands(
        bandsplits: tp.List[tp.Tuple[int, int]],
        sr: int = 44100,
        n_fft: int = 2048
) -> tp.List[tp.Tuple[int, int]]:
    """
    Returns start and end FFT indices of given bandsplits
    """
    freqs = get_fftfreq(sr=sr, n_fft=n_fft)
    band_indices = get_subband_indices(freqs, bandsplits)
    return band_indices


def get_mel_bandwidth_indices(sr, n_fft, n_mels):
    indices = list()

    mel_filter_bank_numpy = filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)
    mel_filter_bank[0][0] = 1
    mel_filter_bank[-1][-1] = 1
    mel_filter_bank_bool = (mel_filter_bank > 0).tolist()

    for mel_filter_bool in mel_filter_bank_bool:
        start_index = np.argmax(mel_filter_bool)
        end_index = len(mel_filter_bool) - np.argmax(mel_filter_bool[::-1])
        indices.append((start_index, end_index))

    return indices


def get_duplicate_indices(bandwidth_indices):
    indices_all = list()
    for bandwidth_index in bandwidth_indices:
        for i in range(bandwidth_index[0], bandwidth_index[1]):
            indices_all.append(i)

    element_count = Counter(indices_all)
    duplicate_indices = [element for element, count in element_count.items() if count > 1]
    return duplicate_indices


if __name__ == '__main__':
    # freqs_splits = [
    #     (1000, 100),
    #     (4000, 250),
    #     (8000, 500),
    #     (16000, 1000),
    #     (20000, 2000),
    # ]
    # sr = 44100
    # n_fft = 2048
    #
    # out = freq2bands(freqs_splits, sr, n_fft)
    #
    # assert sum(out) == n_fft // 2 + 1
    #
    # print(f"Input:\n{freqs_splits}\n{sr}\n{n_fft}\nOutput:{out}")


    n_mels = 32
    sr = 44100
    n_fft = 2048

    indices = get_mel_bandwidth_indices(sr, n_fft, n_mels)

    bandwidths = [(e - s) for s, e in get_mel_bandwidth_indices(sr, n_fft, n_mels)]

    print(indices)
    print(bandwidths)
