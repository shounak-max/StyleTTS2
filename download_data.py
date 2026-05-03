import os
from datasets import load_dataset
import soundfile as sf
import io

print("Loading dataset metadata (no streaming, no decoding)...")
import datasets
# Load dataset but prevent automatic audio decoding to bypass torchaudio/torchcodec requirements
ds = load_dataset('SPRINGLab/IndicTTS_Bengali', split='train', streaming=True).cast_column("audio", datasets.Audio(decode=False))

os.makedirs("Data/wavs", exist_ok=True)

print("Extracting audio files and generating raw text list...")
with open("Data/bn_raw_list.txt", "w", encoding="utf-8") as f:
    for i, item in enumerate(ds):
        if i >= 1000: # Download 1000 samples for training
            break
            
        audio_data = item['audio']['bytes']
        text = item['text'] # Adjust if the column name is different
        
        wav_path = f"Data/wavs/bn_{i:04d}.wav"
        
        # Write binary wav data to file
        with open(wav_path, "wb") as f_wav:
            f_wav.write(audio_data)
            
        # Write to raw list (wav_path|text|speaker_id)
        f.write(f"{wav_path}|{text}|0\n")

print("Done creating Data/bn_raw_list.txt with 100 samples.")
