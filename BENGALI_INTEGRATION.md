# StyleTTS2 Bengali Integration: Architectural Overview

This document outlines the modifications made to the original StyleTTS2 architecture to support the Bengali language, along with performance optimizations and future roadmap.

## 1. Text Frontend & Phonemization
The original English-centric text pipeline was replaced with a robust Bengali IPA (International Phonetic Alphabet) system.

- **Unicode Integration**: Modified `text_utils.py` to include the Bengali Unicode block (`U+0980` - `U+09FF`), allowing native script support.
- **IPA-based G2P**: Implemented `bengali_g2p.py` using `espeak-ng` (language code `bn`) via the `phonemizer` library. This ensures high-fidelity phonetic representation.
- **Pre-processing**: Added a batch phonemization script to convert raw Bengali text into IPA before training, reducing runtime CPU overhead and ensuring token consistency.

## 2. Transfer Learning Strategy
To overcome data scarcity and reduce training time, we shifted from scratch-training to a fine-tuning approach.

- **Cross-Lingual Initialization**: Models are initialized from pre-trained English checkpoints. This leverages universal speech characteristics (prosody, vocal tract modeling) and adapts them to Bengali phonetics.
- **Vocabulary Adaptation**: The `n_token` parameter was tuned to the Bengali phoneme inventory size, and embedding layers were fine-tuned with a lower learning rate to preserve pre-trained knowledge.

## 3. Training & Performance Optimizations
Significant changes were made to handle the high throughput requirements of Bengali training on modern GPUs.

- **Pre-computed Mels**: Introduced `precompute_mels.py` to cache Mel spectrograms on disk. This eliminates the I/O bottleneck caused by real-time STFT calculations.
- **Mixed Precision (BF16)**: Training scripts were updated to use BF16 mixed-precision. This reduces memory footprint and increases training speed on NVIDIA RTX 40-series hardware without sacrificing stability.
- **Parallel Data Loading**: Optimized the data loader with increased workers and pinned memory to ensure the GPU remains fully saturated.

## 4. Alignment & Stability Improvements
- **Monotonic Alignment Shadowing Fix**: Resolved issues in the monotonic aligner (MAS) where the model would skip or repeat phonemes by implementing a custom path calculation in `monotonic_align_stub.py`.
- **Duration Modeling**: Adjusted the duration predictor to better capture the rhythm of Bengali speech, which is more syllable-timed compared to the stress-timed nature of English.

---

## 5. Future Roadmap & Planned Changes

The following enhancements are planned for the next phase of the Bengali StyleTTS2 integration:

### A. Dedicated Bengali Phoneme-BERT
Currently, we use a fine-tuned version of the English PL-BERT. We plan to train or integrate a dedicated Bengali Phoneme-BERT trained on large-scale Bengali corpora to provide better semantic and prosodic context.

### B. Multi-Dialect Support
Expansion of the dataset to include regional Bengali dialects (e.g., Dhakaiya, Chittagonian, Sylheti) to improve the model's versatility across different speech patterns.

### C. Emotional & Expressive Fine-tuning
Leveraging the StyleTTS2 Style Encoder to fine-tune on emotional speech datasets, enabling the model to generate Bengali speech with varied tones (angry, happy, whispered, etc.).

### D. Real-time Inference Optimization
- **ONNX/TensorRT Export**: Optimizing the model for deployment in real-time applications by exporting to high-performance inference formats.
- **Streamlined Vocoder**: Testing lighter versions of the BigVGAN/HiFi-GAN vocoders to reduce latency in streaming TTS applications.

### E. Low-Resource Adaptation
Developing recipes for "Few-Shot" Bengali voice cloning, allowing the creation of a new Bengali voice with as little as 5-10 minutes of audio data.
