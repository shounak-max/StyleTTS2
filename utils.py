try:
    from monotonic_align import maximum_path
    from monotonic_align import mask_from_lens
    from monotonic_align.core import maximum_path_c
    _USE_COMPILED_ALIGN = True
except ImportError:
    # Fallback for environments without C++ build tools (e.g., Windows without WSL)
    from monotonic_align_stub import maximum_path_py as _maximum_path_py
    from monotonic_align_stub import mask_from_lens
    from monotonic_align_stub import maximum_path_c
    _USE_COMPILED_ALIGN = False

import numpy as np
import torch
import copy
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
from munch import Munch

if not _USE_COMPILED_ALIGN:
    # Pure-Python fallback wrapper (slow — only used when C extension unavailable)
    def maximum_path(neg_cent, mask):
        """Python fallback for maximum_path. Install monotonic_align for speed."""
        return _maximum_path_py(neg_cent, mask)
else:
    # Compiled C extension is available — wrap it in the standard interface
    def maximum_path(neg_cent, mask):
        """Cython/C optimized monotonic alignment path."""
        device = neg_cent.device
        dtype = neg_cent.dtype
        neg_cent = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
        path = np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))
        t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
        t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
        maximum_path_c(path, neg_cent, t_t_max, t_s_max)
        return torch.from_numpy(path).to(device=device, dtype=dtype)


def get_data_path_list(train_path=None, val_path=None):
    if train_path is None:
        train_path = "Data/train_list.txt"
    if val_path is None:
        val_path = "Data/val_list.txt"

    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_list = f.readlines()
    with open(val_path, 'r', encoding='utf-8', errors='ignore') as f:
        val_list = f.readlines()

    return train_list, val_list

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

def get_image(arrs):
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(arrs)

    return fig

def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d
    
def log_print(message, logger):
    logger.info(message)
    print(message)
    