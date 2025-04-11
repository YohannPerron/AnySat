import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormHead(nn.Module):
    def __init__(self):
        super(LayerNormHead, self).__init__()

    def forward(self, x):
        """
        Args:
            x: torch.Tensor BxN that contains the logits
        Returns:
            torch.Tensor: LayerNorm applied to x
        """
        return F.layer_norm(x, (x.size(-1),)), 0.0

exp_max_values = {
    torch.float16: 0,
    torch.float32: 50,
    torch.float64: 50,
    torch.bfloat16: 50,
}

def stable_exp(M):
    shift = M.max(dim=-2, keepdim=True).values
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(shift, torch.distributed.ReduceOp.MAX)
    M += exp_max_values[M.dtype] - shift
    return M.exp()


def reduced_sum(*args, **kwargs):
    summed = torch.sum(*args, **kwargs)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(summed)
    return summed

@torch.no_grad()
def sinkhorn_knopp(
    M,
    n_iterations: int,
    eps: float = 1e-8,
):
    M = stable_exp(M)
    for _ in range(n_iterations):
        M /= reduced_sum(M, dim=-2, keepdim=True) + eps
        M /= torch.sum(M, dim=-1, keepdim=True) + eps
    return M


class OnlineClusteringHead(nn.Module):
    """
    Online clustering head for the encoder. See CAPI : https://arxiv.org/pdf/2502.08769

    Implementation from https://github.com/facebookresearch/capi/tree/main
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        bias: bool,  # god why is this bias still here
        n_sk_iter: int,
        target_temp: float,
        pred_temp: float,
        positionwise_sk: bool = True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.n_sk_iter = n_sk_iter
        self.target_temp = target_temp
        self.pred_temp = pred_temp
        self.positionwise_sk = positionwise_sk
        self.layer = nn.Linear(in_dim, out_dim, bias=bias)
        torch.nn.init.normal_(self.layer.weight, std=1)
        if bias:
            torch.nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        b,h,w,c = x.shape
        x = x.view(b, -1, c)
        x_n = nn.functional.normalize(x, dim=-1, p=2, eps=1e-7)
        logits = self.layer(x_n)
        if not self.positionwise_sk:
            logits = logits.flatten(0, -2)
        assignments = sinkhorn_knopp(logits.detach() / self.target_temp, n_iterations=self.n_sk_iter)
        tgt = assignments.flatten(0, -2).float()
        pred = logits.flatten(0, -2).float()
        loss = -torch.sum(tgt * F.log_softmax(pred / self.pred_temp, dim=-1), dim=-1).mean()
        return assignments.detach().view(b,h,w,c), loss
