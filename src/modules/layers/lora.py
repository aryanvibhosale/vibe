import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert isinstance(base, nn.Linear), "LoRALinear only supports wrapping nn.Linear."

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self._base_scaling = alpha / r if r > 0 else 0.0
        
        # 使用 buffer 存储 scaling，这样修改值不会触发 torch.compile 重编译
        # persistent=False 表示不保存到 state_dict，避免加载时 missing key
        self.register_buffer("scaling", torch.tensor(self._base_scaling), persistent=False)

        # 直接持有 weight 和 bias（从原始 Linear 转移过来）
        self.weight = base.weight
        self.bias = base.bias  # 可能是 None

        # LoRA 参数
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 基础 Linear 计算
        result = F.linear(x, self.weight, self.bias)
        if self.r <= 0 or self.lora_A is None:
            return result
        # LoRA: result + dropout(x @ A^T @ B^T) * scaling
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return result + self.dropout(lora_out) * self.scaling

    def reset_lora_parameters(self):
        if self.r > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def set_enabled(self, enabled: bool):
        # 使用 fill_ 原地修改 buffer 值，不会触发重编译
        self.scaling.fill_(self._base_scaling if enabled else 0.0)

    @property
    def enabled(self) -> bool:
        return self.scaling.item() != 0.0


def _get_parent_module(root: nn.Module, name: str) -> Optional[nn.Module]:
    parts = name.split(".")
    if len(parts) == 1:
        return root
    parent = root
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None
        parent = getattr(parent, p)
    return parent


def apply_lora_to_named_linear_modules(
    root: nn.Module,
    *,
    target_submodule_names: list[str],
    r: int,
    alpha: float,
    dropout: float,
) -> None:
    for full_name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        short_name = full_name.split(".")[-1]
        if short_name not in target_submodule_names:
            continue

        parent = _get_parent_module(root, full_name)
        if parent is None:
            continue

        # 用 LoRALinear 替换原始 Linear
        lora_layer = LoRALinear(
            base=module,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(parent, short_name, lora_layer)



