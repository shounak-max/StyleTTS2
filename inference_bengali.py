import torch
import yaml
import os
import sys
from munch import Munch
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer

# StyleTTS2 imports
from models import build_model
from Utils.PLBERT.util import load_plbert
from text_utils import TextCleaner
from bengali_g2p import bengali_to_ipa

# Add the ability to prevent tf import
os.environ['USE_TF'] = '0'

print("Loading Configuration...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = yaml.safe_load(open("Configs/config_bn.yml"))
model_params = Munch(config['model_params'])

print("Building Model...")
model = build_model(model_params, load_only_params=True)

# You will need to change this path to your actual trained checkpoint!
checkpoint_path = "Models/Bengali/epoch_2nd_00100.pth"

if not os.path.exists(checkpoint_path):
    print(f"\n[!] Checkpoint {checkpoint_path} not found.")
    print("This is expected if you haven't finished Stage 2 training yet.")
    print("We will load the model with random weights just to verify the pipeline works.")
else:
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['net'])

model.eval()
model.to(device)

print("Loading Text Cleaner and Bengali BERT...")
textcleaner = TextCleaner()
tokenizer = AutoTokenizer.from_pretrained('sagorsarker/bangla-bert-base')
bert = load_plbert('sagorsarker/bangla-bert-base').to(device)

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def generate_speech(text, output_path="output.wav"):
    print(f"\nGenerating speech for: '{text}'")
    
    # 1. Phonemize
    ipa = bengali_to_ipa(text)
    print(f"IPA: {ipa}")
    
    # 2. Text Encoder Tokens
    tokens = textcleaner(ipa)
    tokens = [0] + tokens + [0] # Add padding
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    input_lengths = torch.LongTensor([tokens.shape[1]]).to(device)
    
    # 3. BERT Tokens
    bert_enc = tokenizer(text, return_tensors='pt')
    bert_tokens = bert_enc['input_ids'].to(device)
    bert_lengths = torch.LongTensor([bert_tokens.shape[1]]).to(device)
    bert_mask = length_to_mask(bert_lengths).to(device)
    
    with torch.no_grad():
        # Get semantic embeddings from BERT
        bert_attn_mask = (~bert_mask).int()
        bert_dur = bert(bert_tokens, attention_mask=bert_attn_mask)
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2) 
        
        # Get acoustic representation
        text_mask = length_to_mask(input_lengths).to(device)
        s_predict = torch.randn(1, 128).to(device) # Random style vector for testing
        
        # Predict duration and pitch
        d, p = model.predictor(d_en, s_predict, input_lengths, text_mask)
        
        # Expand tokens based on duration
        mel_len = int(d.sum().item())
        mel_mask = length_to_mask(torch.LongTensor([mel_len]).to(device)).to(device)
        
        en = model.text_encoder(tokens, input_lengths, text_mask)
        F0_pred, N_pred = model.predictor.F0Ntrain(p, s_predict)
        
        # Length-regulate: expand text encoding by predicted durations
        # Build hard alignment matrix from predicted durations
        dur_rounded = d.squeeze().clamp(min=1).round().long()  # [T_phonemes]
        mel_len_pred = int(dur_rounded.sum().item())
        aln_trg = torch.zeros(dur_rounded.shape[0], mel_len_pred, device=device)
        c_frame = 0
        for t_idx in range(aln_trg.shape[0]):
            dur_t = int(dur_rounded[t_idx].item())
            aln_trg[t_idx, c_frame: c_frame + dur_t] = 1
            c_frame += dur_t

        # Expand text encoding to mel frame resolution via alignment
        en_expanded = en @ aln_trg.unsqueeze(0)  # [1, hidden, mel_len_pred]
        mel_mask_pred = length_to_mask(
            torch.LongTensor([mel_len_pred]).to(device)
        ).to(device)

        # Generate Audio
        out = model.decoder(en_expanded, F0_pred, N_pred, mel_mask_pred.unsqueeze(1).squeeze(2))
        audio = out.squeeze().cpu().numpy()
        
    print(f"Saving to {output_path}...")
    sf.write(output_path, audio, 24000)
    print("Done!")

if __name__ == "__main__":
    test_text = "আমি বাংলায় কথা বলি, বাংলাদেশ আমার দেশ।"
    generate_speech(test_text, "bengali_test_output.wav")
