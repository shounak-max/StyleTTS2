"""
Pre-cache Mel Spectrograms for faster training.

Run this ONCE before training. It reads every .wav file listed in your
training/validation/OOD data lists, computes the mel spectrogram, and
saves it as a .mel.npy file next to the original .wav.

During training, meldataset.py will automatically load .mel.npy if it
exists, skipping the mel computation entirely.

Usage:
    python precompute_mels.py
    python precompute_mels.py --lists Data/bn_train_list.txt Data/bn_val_list.txt
"""
import os
import sys
import argparse
import numpy as np
import soundfile as sf
import librosa
import torch
import torchaudio
from tqdm import tqdm

# Mel spectrogram parameters (must match meldataset.py exactly)
SR = 24000
N_MELS = 80
N_FFT = 2048
WIN_LENGTH = 1200
HOP_LENGTH = 300
LOG_MEAN = -4.0
LOG_STD = 4.0

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=N_MELS, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH
)


def compute_mel(wav_path: str) -> np.ndarray:
    """Load a WAV file and return its normalized log-mel spectrogram."""
    wave, sr = sf.read(wav_path)
    if wave.ndim == 2:
        wave = wave[:, 0]
    if sr != SR:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=SR)

    # Match the padding in meldataset.py _load_tensor
    wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)

    wave_tensor = torch.from_numpy(wave).float()
    mel = to_mel(wave_tensor)  # [n_mels, T]
    mel = (torch.log(1e-5 + mel.unsqueeze(0)) - LOG_MEAN) / LOG_STD  # [1, n_mels, T]
    return mel.squeeze(0).numpy()  # [n_mels, T]


def collect_wav_paths(list_files: list[str]) -> set[str]:
    """Collect all unique wav paths from data list files."""
    paths = set()
    for list_file in list_files:
        if not os.path.exists(list_file):
            print(f"[WARN] {list_file} not found, skipping.")
            continue
        with open(list_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if parts and parts[0]:
                    paths.add(parts[0])
    return paths


def main():
    parser = argparse.ArgumentParser(description="Pre-cache mel spectrograms")
    parser.add_argument(
        "--lists",
        nargs="+",
        default=["Data/bn_train_list.txt", "Data/bn_val_list.txt", "Data/bn_OOD_texts.txt"],
        help="Data list files to process",
    )
    parser.add_argument(
        "--root", default="", help="Root path prefix for wav files (if not absolute)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute even if .mel.npy already exists"
    )
    args = parser.parse_args()

    wav_paths = collect_wav_paths(args.lists)
    print(f"Found {len(wav_paths)} unique wav files to process.")

    skipped = 0
    errors = 0
    computed = 0

    for wav_path in tqdm(sorted(wav_paths), desc="Computing mels"):
        full_path = os.path.join(args.root, wav_path) if args.root else wav_path
        cache_path = full_path.replace(".wav", ".mel.npy")

        if os.path.exists(cache_path) and not args.overwrite:
            skipped += 1
            continue

        if not os.path.exists(full_path):
            print(f"[WARN] WAV not found: {full_path}")
            errors += 1
            continue

        try:
            mel = compute_mel(full_path)
            np.save(cache_path, mel)
            computed += 1
        except Exception as e:
            print(f"[ERROR] {full_path}: {e}")
            errors += 1

    print(f"\nDone. Computed: {computed}, Skipped (cached): {skipped}, Errors: {errors}")
    print(f"\nTo use the cache during training, add this to meldataset.py _load_tensor():")
    print("  cache = wave_path.replace('.wav', '.mel.npy')")
    print("  if os.path.exists(cache):")
    print("      return np.load(cache)  # skip mel computation")


if __name__ == "__main__":
    main()
