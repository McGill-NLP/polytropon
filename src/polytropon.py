import logging
import math
import scipy

import torch
from torch import nn

from adapters import (
    HyperLoRALinear,
    SkilledLoRALinear,
)
from utils import replace_layers, inform_layers


logger = logging.getLogger(__name__)

VARIANT2CLASS = {
    "hyperformer": HyperLoRALinear
}


class SkilledMixin(nn.Module):
    def __init__(
        self,
        model,
        n_tasks: int,
        n_skills: int,
        skilled_variant: str = "learned",
        freeze_pretrained: bool = True,
        custom_skills: str = None,
        state_dict = None,
    ):
        super().__init__()
        self.model = model
        self.n_tasks = n_tasks
        self.n_skills = n_skills
        self.skilled_variant = skilled_variant

        if freeze_pretrained:
            for p in self.model.parameters():
                p.requires_grad = False

        if skilled_variant == "custom":
            self.register_buffer("skill_logits", custom_skills)

        adapter_class = VARIANT2CLASS.get(skilled_variant, SkilledLoRALinear)
        self.adapter_class = adapter_class
        replace_layers(self.model, nn.Linear, adapter_class, n_skills, n_tasks)

        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=False)
            self.model.tie_weights()

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
        inform_layers(self.model, self.adapter_class, skills)
        return self.model.generate(*args, **kwargs)

    def forward(self, task_ids, *args, prior="none", **kwargs):
        skills = self.get_skills(task_ids)
        inform_layers(self.model, self.adapter_class, skills)
        outputs = self.model.forward(*args, **kwargs)

        if self.training and self.skilled_variant == "learned" and prior == "ibp":
            aux_loss = [self.neg_log_IBP(p) for n, p in self.model.named_parameters() if "skill_logits" in n]
            outputs.loss += torch.cat(aux_loss).sum()

        return outputs

    def neg_log_IBP(self, matrix, alpha=3.):
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
        return - logp

if __name__ == "__main__":
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = SkilledMixin(model, n_tasks=2, n_skills=2)
    inputs = ["Tell me, oh Muse, of that ingenious hero who travelled far and wide after he had sacked the famous town of Troy.",
        "Many cities did he visit, and many were the nations with whose manners and customs he was acquainted."]
    inputs = tokenizer(inputs, return_tensors="pt", padding=True)
    task_ids = torch.LongTensor([0, 1])
    logger.warning("forward method: %s", model.forward(task_ids, labels=inputs["input_ids"], **inputs))
    logger.warning("generate method: %s", model.generate(task_ids, **inputs))
