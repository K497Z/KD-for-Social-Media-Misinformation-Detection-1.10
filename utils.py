import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)

def compute_mse_loss(student_f, teacher_f):
    return F.mse_loss(student_f, teacher_f)

def extract_social_emotion(comments: List[str], extractor_func, target_dim=768):
    return torch.zeros(len(comments), target_dim)

def aggregate_features(s_p, s_c, e, weights):
    f = weights.get("w_p", 0.5) * s_p
    if e is not None:
        f += weights.get("w_e", 0.2) * e
    if s_c is not None:
        f += weights.get("w_c", 0.3) * s_c
    return f

def compute_co_attention(P, C, W_l, W_p, W_c, w_hp, w_hc):
    B, d, L = P.shape
    return P.mean(dim=2), C.mean(dim=2), None, None
