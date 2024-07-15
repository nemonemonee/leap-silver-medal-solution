import torch

def r2_score(pred:torch.Tensor, tgt:torch.Tensor) -> float:
    ss_res = torch.sum((tgt - pred) ** 2)
    ss_tot = torch.sum((tgt - torch.mean(tgt)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2