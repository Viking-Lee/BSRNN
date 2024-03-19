import typing as tp
import numpy as np

import torch


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


def freq2bands_generic(
        bandsplits_generic: tp.List[tp.Tuple[int, int]],
        sr: int = 44100,
        n_fft: int = 2048
) -> tp.List[tp.Tuple[int, int]]:
    """
    :param bandsplits_generic: [(end_freq, step)]
    :param sr: sampliing rate
    :param n_fft:
    :return: subbands freq bins
    """
    freqs = sr * torch.fft.fftfreq(n_fft)[: n_fft //2 + 1]
    freqs[-1] = sr // 2

    indices = []
    start_freq, start_index = 0, 0

    for end_freq, step in bandsplits_generic:
        bands = torch.arange(start_freq + step, end_freq + step, step)
        start_freq = end_freq
        for band in bands:
            end_index = freqs[freqs < band].shape[0]
            if end_index != start_index or not True:
                indices.append((start_index, end_index))
            start_index = end_index
    indices.append((start_index, freqs.shape[0]))
    return indices


def get_microtone_name(semitones_from_A4, divisions_per_octave):
    """get musical note name from freq"""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    microtone_index = (
        int((semitones_from_A4 + 9) * divisions_per_octave / 12) % divisions_per_octave
    )
    octave = int((semitones_from_A4 + 9) // 12)
    note_index = microtone_index // (divisions_per_octave // 12)
    microtone_suffix = f"+{microtone_index % (divisions_per_octave // 12)}"
    return (
        notes[note_index]
        + (microtone_suffix if microtone_suffix != "+0" else "")
        + str(octave)
    )


def evenly_skip_elements(input_list, k):
    step = len(input_list) / k
    new_list = [input_list[int(i * step)] for i in range(k)]

    return new_list


def microtonal_notes(divisions_per_octave):
    A4_freq = 440.0
    min_freq = 20.0
    max_freq = 20000.0
    notes = []
    freqs = []
    current_freq = min_freq
    while current_freq <= max_freq:
        semitones_from_A4 = 12 * np.log2(current_freq / A4_freq)
        nearest_microtone = round(semitones_from_A4 * divisions_per_octave / 12)
        nearest_freq = A4_freq * (2 ** (nearest_microtone / divisions_per_octave))

        # Check if the frequency is a microtonal (not a standard semitone)
        if nearest_microtone % (divisions_per_octave // 12) != 0:
            note_name = get_microtone_name(
                nearest_microtone / (divisions_per_octave / 12), divisions_per_octave
            )
            notes.append(note_name)
            freqs.append(nearest_freq)

        # Move to the next microtone
        current_freq = A4_freq * (2 ** ((nearest_microtone + 1) / divisions_per_octave))
    return (notes, freqs)

def create_evenly_distributed_splits(n_splits):
    _, freqs = microtonal_notes(24)
    splits = []
    last_freq = 0
    for freq in evenly_skip_elements(freqs, n_splits):
        splits.append((freq, freq - last_freq))
        last_freq = freq
    return splits


if __name__ == '__main__':
    freqs_splits = [
        (1000, 100),
        (4000, 250),
        (8000, 500),
        (16000, 1000),
        (20000, 2000),
    ]
    sr = 44100
    n_fft = 2048

    out = freq2bands(freqs_splits, sr, n_fft)

    # assert sum(out) == n_fft // 2 + 1

    n_subbands = 41
    splits_generic = create_evenly_distributed_splits(n_subbands)
    out_generic = freq2bands_generic(splits_generic, sr, n_fft)


    print(f"Input:\n{splits_generic}\n{sr}\n{n_fft}\nOutput:{out_generic}")
