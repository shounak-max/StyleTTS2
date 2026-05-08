"""
StyleTTS2 Bengali — Quick Sanity Test
Verifies every major component without needing training data or a checkpoint.
Run: python test_sanity.py

Green = working. Red = broken. Fix reds before training.
"""
import sys, time, traceback
import torch
import numpy as np

PASS = "\033[92m  [PASS]\033[0m"
FAIL = "\033[91m  [FAIL]\033[0m"
WARN = "\033[93m  [WARN]\033[0m"
SKIP = "\033[94m  [SKIP]\033[0m"

results = []

def test(name, fn):
    t0 = time.time()
    try:
        note = fn()
        elapsed = time.time() - t0
        tag = PASS
        results.append((name, True, note or "", elapsed))
        print(f"{tag} {name:<45} {note or '':<35} ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        results.append((name, False, str(e)[:80], elapsed))
        print(f"{FAIL} {name:<45} {str(e)[:80]}")

print("\n" + "="*80)
print("  StyleTTS2 Bengali — Sanity Test")
print("="*80 + "\n")

# ─── 1. CUDA ──────────────────────────────────────────────────────────────────
def check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — check NVIDIA drivers")
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    cap = torch.cuda.get_device_capability(0)
    bf16_ok = cap[0] >= 8  # BF16 needs Ampere (sm_80) or newer
    bf16_note = "BF16 supported" if bf16_ok else f"BF16 NOT supported (sm_{cap[0]}{cap[1]}, need sm_80+) — use fp16"
    return f"{name} | {vram:.1f} GB VRAM | {bf16_note}"

test("CUDA availability", check_cuda)

# ─── 2. PyTorch version ───────────────────────────────────────────────────────
def check_torch():
    v = torch.__version__
    if int(v.split(".")[0]) < 2:
        raise RuntimeError(f"PyTorch {v} — upgrade to 2.x for torch.compile support")
    return f"PyTorch {v}"

test("PyTorch version", check_torch)

# ─── 3. Symbol table (text_utils) ────────────────────────────────────────────
def check_symbols():
    from text_utils import symbols, dicts, TextCleaner
    tc = TextCleaner()
    # Test Bengali characters
    bn_sample = "আমি বাংলায় কথা বলি"
    ids = tc(bn_sample)
    if len(ids) == 0:
        raise RuntimeError("Bengali chars not in symbol table — check text_utils.py")
    # Test IPA characters
    ipa_sample = "aɪ wɪl tɛst"
    ids_ipa = tc(ipa_sample)
    return f"{len(symbols)} symbols | Bengali: {len(ids)} ids | IPA: {len(ids_ipa)} ids"

test("Symbol table (text_utils)", check_symbols)

# ─── 4. meldataset imports ────────────────────────────────────────────────────
def check_meldataset():
    from meldataset import FilePathDataset, build_dataloader, TextCleaner as TC2
    from text_utils import TextCleaner as TC1
    # Verify they're the same class (deduplication worked)
    return f"Import OK | TextCleaner deduplicated: {TC1 is TC2}"

test("meldataset imports", check_meldataset)

# ─── 5. maximum_path (compiled vs stub) ───────────────────────────────────────
def check_maximum_path():
    from utils import maximum_path, _USE_COMPILED_ALIGN
    # Run a tiny alignment to verify it actually works
    B, T_mel, T_text = 2, 50, 10
    neg_cent = torch.randn(B, T_mel, T_text)
    mask = torch.ones(B, T_mel, T_text)
    path = maximum_path(neg_cent, mask)
    assert path.shape == (B, T_mel, T_text), "Wrong output shape"
    mode = "C extension (FAST)" if _USE_COMPILED_ALIGN else "Python stub (SLOW — compile monotonic_align for TMA)"
    return mode

test("maximum_path alignment", check_maximum_path)

# ─── 6. G2P / espeak-ng ───────────────────────────────────────────────────────
def check_g2p():
    try:
        from phonemizer import phonemize
        result = phonemize("আমি বাংলায় কথা বলি", language="bn", backend="espeak",
                          with_stress=True, language_switch='remove-flags')
        if not result or not result.strip():
            raise RuntimeError("Empty output from phonemizer")
        return f"IPA: '{result.strip()[:40]}'"
    except Exception as e:
        if "espeak" in str(e).lower():
            raise RuntimeError("espeak-ng not found — install from https://github.com/espeak-ng/espeak-ng/releases")
        raise

test("G2P Bengali → IPA (espeak-ng)", check_g2p)

# ─── 7. BanglaBERT ────────────────────────────────────────────────────────────
def check_banglabert():
    from transformers import AutoTokenizer, AutoModel
    model_name = 'sagorsarker/bangla-bert-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    
    # Tiny forward pass
    text = "আমি বাংলায় কথা বলি"
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        out = model(**inputs)
    hidden = out.last_hidden_state
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return f"{params:.0f}M params | output shape: {tuple(hidden.shape)}"

test("BanglaBERT forward pass", check_banglabert)

# ─── 8. XLSR-53 / wav2vec2-base SLM ─────────────────────────────────────────
def check_slm():
    from transformers import Wav2Vec2Model
    # Use the fast small model from config_bn_fast.yml
    model_name = 'facebook/wav2vec2-base'
    model = Wav2Vec2Model.from_pretrained(model_name).eval()
    
    # Dummy 1-second audio at 16kHz
    dummy_audio = torch.randn(1, 16000)
    with torch.no_grad():
        out = model(dummy_audio)
    hidden = out.last_hidden_state
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return f"{params:.0f}M params | output shape: {tuple(hidden.shape)}"

test("SLM wav2vec2-base forward pass", check_slm)

# ─── 9. Full model build ──────────────────────────────────────────────────────
def check_model_build():
    import yaml
    from munch import Munch
    from utils import recursive_munch
    from models import build_model, load_ASR_models, load_F0_models
    from Utils.PLBERT.util import load_plbert
    
    config = yaml.safe_load(open('Configs/config_bn.yml'))
    model_params = recursive_munch(config['model_params'])
    
    # Load sub-models
    text_aligner = load_ASR_models(config['ASR_path'], config['ASR_config'])
    pitch_extractor = load_F0_models(config['F0_path'])
    bert = load_plbert(config['PLBERT_dir'])
    
    # Build full model (no checkpoint needed)
    model = build_model(model_params, text_aligner, pitch_extractor, bert)
    
    total_params = sum(
        sum(p.numel() for p in model[k].parameters()) for k in model
    ) / 1e6
    return f"All sub-models built | Total: {total_params:.0f}M params"

test("Full model build (no checkpoint)", check_model_build)

# ─── 10. GPU dummy forward pass ───────────────────────────────────────────────
def check_gpu_forward():
    device = 'cuda'
    # Simulate what train_first.py does: mel encoder forward
    import yaml
    from munch import Munch
    from utils import recursive_munch
    from models import build_model, load_ASR_models, load_F0_models
    from Utils.PLBERT.util import load_plbert
    
    config = yaml.safe_load(open('Configs/config_bn.yml'))
    model_params = recursive_munch(config['model_params'])
    text_aligner = load_ASR_models(config['ASR_path'], config['ASR_config'])
    pitch_extractor = load_F0_models(config['F0_path'])
    bert = load_plbert(config['PLBERT_dir'])
    model = build_model(model_params, text_aligner, pitch_extractor, bert)
    
    # Move to GPU
    _ = [model[k].to(device).eval() for k in model]
    
    # Dummy batch (B=2, 80 mel bins, 200 frames)
    B, n_mels, T = 2, 80, 200
    T_text = 30
    mel = torch.randn(B, n_mels, T).to(device)
    texts = torch.randint(0, 418, (B, T_text)).to(device)
    input_lengths = torch.LongTensor([T_text, T_text - 5]).to(device)
    mel_input_length = torch.LongTensor([T, T - 20]).to(device)
    
    with torch.no_grad():
        mask = torch.zeros(B, T // 4).bool().to(device)
        _, _, attn = model.text_aligner(mel, mask, texts)
    
    vram_used = torch.cuda.memory_allocated(0) / 1e9
    return f"GPU forward pass OK | VRAM used: {vram_used:.2f} GB"

test("GPU forward pass (dummy batch)", check_gpu_forward)

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*80)
passed = sum(1 for _, ok, _, _ in results if ok)
total = len(results)
print(f"\n  Results: {passed}/{total} passed\n")

critical_fails = [name for name, ok, _, _ in results if not ok and name in [
    "CUDA availability", "Symbol table (text_utils)", "Full model build (no checkpoint)",
    "GPU forward pass (dummy batch)", "BanglaBERT forward pass"
]]

if critical_fails:
    print(f"\033[91m  CRITICAL failures (fix before training):\033[0m")
    for f in critical_fails:
        print(f"    - {f}")
else:
    print(f"\033[92m  All critical checks passed. Ready to train.\033[0m")

non_critical = [name for name, ok, _, _ in results if not ok and name not in critical_fails]
if non_critical:
    print(f"\033[93m  Non-critical (training will work but slower):\033[0m")
    for f in non_critical:
        print(f"    - {f}")

print()
