"""
Bengali G2P utility for StyleTTS2.
Converts Bengali text to IPA phoneme sequences using espeak-ng Bengali backend.

Prerequisites:
    pip install phonemizer
    # espeak-ng must be installed with Bengali support

Usage:
    # As a module:
    from bengali_g2p import bengali_to_ipa
    ipa = bengali_to_ipa("আমি বাংলায় কথা বলি")

    # Batch pre-processing of data list files:
    python bengali_g2p.py --input Data/bn_raw_list.txt --output Data/bn_train_list.txt
    
    # The input file format should be: wavpath|bengali_text|speaker_id
    # The output file will be:         wavpath|ipa_phonemes|speaker_id
"""
import argparse
import sys
import os

# Add eSpeak NG to PATH (required for phonemizer on Windows)
espeak_path = r"C:\Program Files\eSpeak NG"
if espeak_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] += os.pathsep + espeak_path
    
# Set PHONEMIZER_ESPEAK_LIBRARY specifically for phonemizer on Windows to find the dll
# The library is located in the lib folder for espeak-ng Windows builds.
espeak_lib = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
if os.path.exists(espeak_lib):
    os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = espeak_lib

from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator


# Initialize the Bengali espeak backend
# This uses espeak-ng's built-in Bengali (bn) phonemizer
_backend = EspeakBackend(
    language='bn',           # Bengali language code
    preserve_punctuation=True,
    with_stress=False,
    language_switch='remove-flags'  # remove language switch flags for clean IPA
)

def bengali_to_ipa(text: str) -> str:
    """
    Convert Bengali text to IPA phoneme string using espeak-ng.
    
    Args:
        text: Bengali text string (Unicode)
    
    Returns:
        IPA phoneme string with space-separated phones
    """
    if not text or not text.strip():
        return ""
    
    result = _backend.phonemize(
        [text],
        strip=True
    )
    return result[0] if result else ""


def bengali_to_ipa_batch(texts: list) -> list:
    """
    Convert a batch of Bengali text strings to IPA.
    
    Args:
        texts: List of Bengali text strings
    
    Returns:
        List of IPA phoneme strings
    """
    if not texts:
        return []
    
    return _backend.phonemize(
        texts,
        strip=True
    )


def preprocess_file(input_path: str, output_path: str):
    """
    Pre-phonemize an entire data list file from Bengali text to IPA.
    
    Input format:  wavpath|bengali_text|speaker_id
    Output format: wavpath|ipa_phonemes|speaker_id
    
    Args:
        input_path: Path to input data list file
        output_path: Path to output phonemized data list file
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed = []
    errors = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('|')
        if len(parts) < 2:
            print(f"WARNING: Skipping malformed line {i+1}: {line}")
            errors += 1
            continue
            
        wav_path = parts[0]
        text = parts[1]
        speaker_id = parts[2] if len(parts) > 2 else '0'
        
        try:
            ipa = bengali_to_ipa(text)
            if not ipa:
                print(f"WARNING: Empty IPA output for line {i+1}: {text}")
                errors += 1
                continue
            processed.append(f"{wav_path}|{ipa}|{speaker_id}")
        except Exception as e:
            print(f"ERROR processing line {i+1}: {e}")
            errors += 1
            continue

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(processed) + '\n')
    
    print(f"Processed {len(processed)} lines, {errors} errors")
    print(f"Output written to: {output_path}")


def print_phoneme_inventory(input_path: str):
    """
    Scan a phonemized data list file and print all unique phonemes found.
    Useful for computing the n_token config value.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    all_chars = set()
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) >= 2:
            for char in parts[1]:
                all_chars.add(char)
    
    print(f"Total unique characters/phones: {len(all_chars)}")
    print(f"Characters: {sorted(all_chars)}")
    return all_chars


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bengali G2P preprocessor for StyleTTS2')
    parser.add_argument('--input', required=True, help='Input data list file (wavpath|bengali_text|speaker_id)')
    parser.add_argument('--output', required=True, help='Output phonemized data list file')
    parser.add_argument('--inventory', action='store_true', help='Print phoneme inventory from output file')
    
    args = parser.parse_args()
    
    preprocess_file(args.input, args.output)
    
    if args.inventory:
        print("\n--- Phoneme Inventory ---")
        print_phoneme_inventory(args.output)
