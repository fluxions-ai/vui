import torch
from torch import Tensor


def multinomial(input: Tensor, num_samples: int, replacement=False, *, generator=None):
    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(
        input_, num_samples=num_samples, replacement=replacement, generator=generator
    )
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def sample_top_k(probs: Tensor, k: int) -> Tensor:
    top_k_value, _ = torch.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs *= (probs >= min_value_top_k).float()
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs, num_samples=1)
    return next_token


def sample_top_p(probs: Tensor, p: float) -> Tensor:
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token


def sample_top_p_top_k(probs: Tensor, p: float, top_k: int):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = sample_top_k(probs_sort, top_k)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
