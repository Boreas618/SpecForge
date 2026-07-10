import torch

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0


class BF16Optimizer:
    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        offload_master=False,
    ):
        # TODO: For now, we only support cosine annealing warmup lr scheduler and AdamW optimizer
        # TODO: We should make these parameters configurable
        #   These magic numbers: weight_decay=0.0, max_grad_norm=0.5, total_steps=800k, warmup_steps=12k are copied from
        #   https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/ds_config.json
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        # Keep the fp32 master + AdamW moment buffers on CPU to free ~12 bytes/param
        # of GPU memory (needed to fit very large sharded drafts alongside a
        # resident target engine). Only the per-step grad copy + weight write-back
        # cross the PCIe boundary; forward/backward stay fully on GPU.
        self.offload_master = offload_master
        master_device = "cpu" if offload_master else None
        self.fp32_params = [
            (
                p.detach().to("cpu", torch.float32).clone()
                if offload_master
                else p.detach().clone().to(torch.float32)
            )
            for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.optimizer = torch.optim.AdamW(
            self.fp32_params, lr=lr, weight_decay=weight_decay
        )
        self.last_grad_norm = None
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

    def step(self, lr_scale: float = 1.0):
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                if p.grad is None:
                    mp.grad = None
                else:
                    g = p.grad.detach().to(torch.float32)
                    mp.grad = g.to("cpu") if self.offload_master else g
        grad_norm = torch.nn.utils.clip_grad_norm_(self.fp32_params, self.max_grad_norm)
        self.last_grad_norm = grad_norm.detach()
        # lr_scale: transient LR damp for THIS step only (post-resume moment
        # re-warm). The original lr must be restored BEFORE scheduler.step():
        # torch schedulers like CosineAnnealingLR are recurrences on
        # param_group["lr"], so a damped value left in place would permanently
        # drag the whole remaining schedule down.
        if lr_scale != 1.0:
            _orig_lrs = [g["lr"] for g in self.optimizer.param_groups]
            for g in self.optimizer.param_groups:
                g["lr"] = g["lr"] * lr_scale
        self.optimizer.step()
        if lr_scale != 1.0:
            for g, _lr in zip(self.optimizer.param_groups, _orig_lrs):
                g["lr"] = _lr
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(p.device, p.dtype))
                p.grad = None
        return self.last_grad_norm

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        print_on_rank0("Successfully loaded optimizer state_dict.")
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded scheduler state_dict.")

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]
