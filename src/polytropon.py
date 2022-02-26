import logging
import math
import scipy

import torch
from torch import nn
from torch.distributions.beta import Beta

from adapters import (
    HyperLoRALinear,
    SkilledLoRALinear,
)
from utils import replace_layers, inform_layers


logger = logging.getLogger(__name__)

VARIANT2CLASS = {
    "hyperformer": HyperLoRALinear
}


class SkilledMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        n_tasks: int,
        n_skills: int,
        skilled_variant: str = "learned",
        custom_skills: str = None,
        freeze_pretrained: bool = True,
        **kwargs,
    ):
        state_dict = kwargs.pop("state_dict", None)
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if freeze_pretrained:
            for p in model.parameters():
                p.requires_grad = False

        adapter_class = VARIANT2CLASS.get(skilled_variant, SkilledLoRALinear)
        replace_layers(model, nn.Linear, adapter_class, n_skills, n_tasks)
        model.set_extra_params(model, n_tasks, n_skills, skilled_variant, custom_skills, adapter_class)

        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
            model.tie_weights()

        return model

    def set_extra_params(self, n_tasks, n_skills, skilled_variant, custom_skills, adapter_class):
        self.n_tasks = n_tasks
        self.n_skills = n_skills
        self.skilled_variant = skilled_variant
        self.adapter_class = adapter_class

        if skilled_variant == "custom":
            self.register_buffer("skill_logits", custom_skills)

    def get_skills(self, task_ids):
        if self.skilled_variant in ["learned", "hyper"]:
            # skills are computed inside each module
            skills = task_ids
        elif self.skilled_variant == "shared":
            skills = torch.ones((task_ids.size(0), 1), device=task_ids.device)
        elif self.skilled_variant == "private":
            skills = torch.eye(self.n_skills, self.n_skills, device=task_ids.device)[task_ids]
        elif self.skilled_variant == "custom":
            skills = self.skill_logits.to(self.device)[task_ids]
        else:
            raise ValueError

        return skills

    def generate(self, task_ids, *args, **kwargs):
        skills = self.get_skills(task_ids)
        inform_layers(self, self.adapter_class, skills)
        return super().generate(*args, **kwargs)

    def forward(self, task_ids, *args, prior="none", **kwargs):
        skills = self.get_skills(task_ids)
        inform_layers(self, self.adapter_class, skills)
        outputs = super().forward(*args, **kwargs)
        if kwargs.get("is_training", False) and self.skilled_variant == "learned":
            if prior == "beta":
                aux_loss = - Beta(skills.new_full((1,), 1. / self.n_skills),
                                  skills.new_full((1,), 1.)
                                  ).log_prob(skills).sum()
            elif prior == "ibp":
                aux_loss = - self.log_IBP(skills)
            elif prior == "none":
                aux_loss = 0.
            return outputs + aux_loss
        return outputs

    def log_IBP(self, matrix, alpha=3.):
        """ Calculate IBP prior contribution log P(Z|alpha)
            Based on https://github.com/davidandrzej/PyIBP/blob/master/PyIBP.py """
        N, _ = matrix.shape
        m = matrix.sum(dim=0)
        m = m[m.nonzero()].squeeze()
        K = m.shape
        def log_factorial(value):
            return torch.lgamma(value + 1)
        logp = 0.
        logp += K * math.log(alpha)

        for n in range(N):
            new_features = torch.clamp(matrix[n] - matrix.sum(0), min=0., max=1.).sum()
            logp -= log_factorial(new_features)

        logp -= alpha * sum([float(1) / i for i in range(1, N + 1)])
        logp += (log_factorial(N - m) + log_factorial(m - 1)).sum()
        logp -= scipy.special.gammaln(N + 1) * K
        return logp
