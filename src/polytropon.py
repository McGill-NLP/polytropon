import logging
import math
import scipy.special as sps

import torch
from torch import nn
from torch.distributions.beta import Beta
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from transformers import AutoModel

from adapters import HyperLoRALinear, SkilledLoRALinear, EPS
from utils import replace_layers, inform_layers

logger = logging.getLogger(__name__)


class SkilledMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        n_tasks: int,
        n_skills: int,
        skilled_variant: str,
        finegrained: bool = False,
        custom_skills: str = None,
        **kwargs,
    ):
        state_dict = kwargs.pop("state_dict", None)
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        for p in model.parameters():
            p.requires_grad = False

        new_linear = HyperLoRALinear if skilled_variant == "hyper" else SkilledLoRALinear
        replace_layers(model, nn.Linear, new_linear, n_skills, n_tasks if finegrained else 0)
        model.set_extra_params(n_tasks, n_skills, skilled_variant, finegrained, custom_skills)

        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)
            model.tie_weights()

        return model

    def set_extra_params(self, n_tasks, n_skills, skilled_variant, finegrained, custom_skills):
        self.n_skills = n_skills
        self.skilled_variant = skilled_variant
        self.finegrained = finegrained

        if skilled_variant == "learned" and not self.finegrained:
            self.skill_logits = nn.Parameter(torch.empty((n_tasks, n_skills)).uniform_(-1e-3, 1e-3))
        elif skilled_variant == "custom":
            self.register_buffer("skill_logits", custom_skills)

    def get_skills(self, task_ids):
        if self.finegrained:
            # skills are computed inside each module
            skills = task_ids
        else:
            if self.skilled_variant == "learned":
                skill_logits = self.skill_logits[task_ids]
                if self.training:
                    skill_logits = RelaxedBernoulli(temperature=1., logits=skill_logits).rsample()
                else:
                    skill_logits = torch.sigmoid(skill_logits)
                skills = skill_logits / (skill_logits.sum(dim=-1, keepdim=True) + EPS)
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
        new_linear = HyperLoRALinear if self.skilled_variant == "hyper" else SkilledLoRALinear
        inform_layers(self, new_linear, skills)
        return super().generate(*args, **kwargs)

    def skilled_forward(self, task_ids, *args, prior="none", **kwargs):
        skills = self.get_skills(task_ids)
        new_linear = HyperLoRALinear if self.skilled_variant == "hyper" else SkilledLoRALinear
        inform_layers(self, new_linear, skills)
        outputs = self.forward(*args, **kwargs)
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
        logp -= sps.gammaln(N + 1) * K
        return logp


class Polytropon(SkilledMixin, AutoModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
