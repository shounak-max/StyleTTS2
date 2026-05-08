import os
import yaml
import torch
from transformers import AutoModel, AutoConfig


class BengaliBERT(torch.nn.Module):
    """
    Wrapper around a HuggingFace Bengali BERT model to match the PL-BERT interface.
    
    The original PL-BERT was a custom ALBERT model trained on English phoneme MLM.
    This replaces it with a pretrained Bengali BERT that understands Bengali text
    semantics, providing meaningful embeddings for the duration predictor and 
    style diffusion model.
    
    The forward() method returns only last_hidden_state to match CustomAlbert behavior.
    """
    def __init__(self, model_name='sagorsarker/bangla-bert-base'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        
    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Args:
            input_ids: [B, T] token IDs from BERT tokenizer
            attention_mask: [B, T] attention mask (1 = attend, 0 = pad)
        
        Returns:
            last_hidden_state: [B, T, hidden_size] contextual embeddings
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


def load_plbert(log_dir_or_model_name):
    """
    Load a BERT model for StyleTTS2.
    
    For Bengali: pass a HuggingFace model name like 'sagorsarker/bangla-bert-base'.
    For English (legacy): pass the local PL-BERT directory path containing config.yml
                          and step_*.t7 checkpoint files.
    
    Args:
        log_dir_or_model_name: Either a HuggingFace model name/path or a local
                               directory containing PL-BERT config.yml + checkpoint.
    
    Returns:
        A BERT model instance with .config attribute and forward() returning
        last_hidden_state [B, T, hidden_size].
    """
    # Check if it's a local PL-BERT directory (legacy English mode)
    config_path = os.path.join(log_dir_or_model_name, "config.yml")
    if os.path.isdir(log_dir_or_model_name) and os.path.exists(config_path):
        # Legacy mode: load custom ALBERT from local checkpoint
        return _load_legacy_plbert(log_dir_or_model_name, config_path)
    
    # HuggingFace mode: load Bengali BERT (or any HF model)
    print(f"Loading Bengali BERT from HuggingFace: {log_dir_or_model_name}")
    model = BengaliBERT(model_name=log_dir_or_model_name)
    print(f"  hidden_size={model.config.hidden_size}, "
          f"max_position_embeddings={model.config.max_position_embeddings}")
    return model


def _load_legacy_plbert(log_dir, config_path):
    """
    Load the original English PL-BERT from a local directory.
    Kept for backward compatibility.
    """
    from transformers import AlbertConfig, AlbertModel
    
    class CustomAlbert(AlbertModel):
        def forward(self, *args, **kwargs):
            outputs = super().forward(*args, **kwargs)
            return outputs.last_hidden_state

    plbert_config = yaml.safe_load(open(config_path))
    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    files = os.listdir(log_dir)
    ckpts = []
    for f in os.listdir(log_dir):
        if f.startswith("step_"): ckpts.append(f)

    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
    iters = sorted(iters)[-1]

    checkpoint = torch.load(log_dir + "/step_" + str(iters) + ".t7", map_location='cpu', weights_only=False)
    state_dict = checkpoint['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:] # remove `encoder.`
            new_state_dict[name] = v
    del new_state_dict["embeddings.position_ids"]
    bert.load_state_dict(new_state_dict, strict=False)
    
    return bert
