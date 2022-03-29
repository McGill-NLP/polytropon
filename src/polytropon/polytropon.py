import logging
import math
from scipy import special

import torch
from torch import nn

from .adapters import (
    HyperLoRALinear,
    SkilledLoRALinear,
    SkilledLTSFTLinear,
)
from .utils import replace_layers, inform_layers


logger = logging.getLogger(__name__)

VARIANT2CLASS = {
    "hyperformer": (HyperLoRALinear, True),
    "sparse": (SkilledLTSFTLinear, False),
}


class SkilledMixin(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        n_tasks: int,
        n_skills: int,
        skilled_variant: str = "learned",
        freeze: bool = True,
        custom_skills: str = None,
        state_dict = None,
    ):
        super().__init__()
        self.model = model
        self.n_tasks = n_tasks
        self.n_skills = n_skills
        self.skilled_variant = skilled_variant

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

        adapter_class, only_attention = VARIANT2CLASS.get(skilled_variant, (SkilledLoRALinear, True))
        self.adapter_class = adapter_class
        skills = self.get_skills(custom_skills)
        replace_layers(self.model, adapter_class, n_tasks, n_skills, skills, only_attention=only_attention)

        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=False)
            self.model.tie_weights()

    def get_skills(self, custom_skills):
        if self.skilled_variant in ["learned", "hyper", "sparse"]:
            # skills are computed inside each module
            skills = None
        elif self.skilled_variant == "shared":
            skills = torch.ones((self.n_tasks, 1), device=task_ids.device)
        elif self.skilled_variant == "private":
            skills = torch.eye(self.n_tasks, self.n_tasks, device=task_ids.device)
        elif self.skilled_variant == "custom":
            skills = custom_skills
        else:
            raise ValueError

        return skills

    def generate(self, task_ids, *args, **kwargs):
        inform_layers(self.model, self.adapter_class, task_ids)
        return self.model.generate(*args, **kwargs)

    def forward(self, task_ids, *args, add_prior=False, **kwargs):
        inform_layers(self.model, self.adapter_class, task_ids)
        outputs = self.model.forward(*args, **kwargs)

        if self.training and self.skilled_variant == "learned" and add_prior:
            aux_loss = [self.neg_log_IBP(p) for n, p in self.model.named_parameters() if "skill_logits" in n]
            outputs.loss += torch.stack(aux_loss).sum()

        return outputs

    def neg_log_IBP(self, matrix, alpha=3.):
        """ Calculate IBP prior contribution - log P(Z|alpha)
            Based on https://github.com/davidandrzej/PyIBP/blob/master/PyIBP.py """
        matrix = torch.sigmoid(matrix)
        matrix_hard = (matrix > .5).float()
        matrix = matrix_hard - matrix.detach() + matrix

        N, _ = matrix.shape
        m = matrix.sum(dim=0)
        m = m[m.nonzero()].squeeze()
        K = len(m)
        def log_factorial(value):
            return torch.lgamma(value + 1)
        logp = 0.
        logp += K * math.log(alpha)

        for n in range(N):
            new_features = torch.clamp(matrix[n] - matrix.sum(0), min=0., max=1.).sum()
            logp -= log_factorial(new_features)

        logp -= alpha * sum([float(1) / i for i in range(1, N + 1)])
        logp += (log_factorial(N - m) + log_factorial(m - 1)).sum()
        logp -= special.gammaln(N + 1) * K
        return - logp


if __name__ == "__main__":
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    inputs = ["Tell me, oh Muse, of that ingenious hero who travelled far and wide after he had sacked the famous town of Troy.",
        "Many cities did he visit, and many were the nations with whose manners and customs he was acquainted."]
    inputs = tokenizer(inputs, return_tensors="pt", padding=True)
    task_ids = torch.LongTensor([0, 1])

    for skilled_variant in ["learned", "hyper", "sparse", "shared", "private"]:
        skilled_model = SkilledMixin(model, n_tasks=2, n_skills=2, skilled_variant=skilled_variant)
        logger.warning("forward %s: %s", skilled_variant, skilled_model.forward(task_ids, labels=inputs["input_ids"], add_prior=True, **inputs))
        logger.warning("generate %s: %s", skilled_variant, skilled_model.generate(task_ids, **inputs))
