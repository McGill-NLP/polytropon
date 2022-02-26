import math
import random
import itertools
from typing import Optional

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.nn.init import calculate_gain
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


EPS = 1e-12


class SkilledModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._task_ids = None

    @property
    def task_ids(self):
        return self._task_ids

    @task_ids.setter
    def task_ids(self, value):
        self._task_ids = value


class HyperLoRALinear(SkilledModule):
    """ Applies a linear function parameterised by a base bias
    and a weighted average of base and task-conditioned weights
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self,
                 n_tasks: int,
                 n_skills: int,
                 skills: Optional[Tensor],
                 weight: Tensor,
                 bias: Optional[Tensor],
                 r: int = 16,
                 freeze: bool = True
                 ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.r = r

        self.task_embs = nn.Embedding(n_tasks, n_skills)
        self.task_proj = nn.Sequential(
            nn.Linear(n_skills, n_skills),
            nn.ReLU(),
            nn.Linear(n_skills, n_skills),
        )

        self.weight = nn.Parameter(weight.data)
        self.weight.requires_grad = not freeze

        self.hyper_weight_A = nn.Linear(n_skills, r * self.in_features, bias=False)
        self.hyper_weight_B = nn.Linear(n_skills, self.out_features * r, bias=False)
        self.scaling = 1 / self.r

        if bias is not None:
            self.bias = nn.Parameter(bias.data)
            self.bias.requires_grad = not freeze
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.hyper_weight_B.weight)

    def forward(self, input: Tensor) -> Tensor:
        # Provisions for inputs repeated for generation
        assert input.size()[0] % self.task_ids.size(0) == 0
        repeats = input.size()[0] // self.task_ids.size(0)
        if repeats > 1:
            self.task_ids = torch.repeat_interleave(self.task_ids, repeats, dim=0)

        task_embs = self.task_embs(self.task_ids)
        task_embs = self.task_proj(task_embs)

        hyper_weight_A = self.hyper_weight_A(task_embs).view(input.size()[0], self.in_features, self.r)
        hyper_weight_B = self.hyper_weight_B(task_embs).view(input.size()[0], self.r, self.out_features)
        output = torch.matmul(input, hyper_weight_A) # bsi,bir->bsr
        output = torch.matmul(output, hyper_weight_B) # bsr,bro->bso
        output = F.linear(input, self.weight, self.bias) + output * self.scaling

        return output


class SkilledLoRALinear(SkilledModule):
    """ Applies a linear function parameterised by a base bias
    and a weighted average of base and skill weights
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self,
                 n_tasks: int,
                 n_skills: int,
                 skills: Optional[Tensor],
                 weight: Tensor,
                 bias: Optional[Tensor],
                 r: int = 16,
                 freeze: bool = True
                 ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.r = r

        if skills is None:
            self.skill_logits = nn.Parameter(torch.empty((n_tasks, n_skills)).uniform_(-1e-3, 1e-3))
            self.is_learned = True
        else:
            self.register_buffer("skill_logits", skills)
            self.is_learned = False

        self.weight = nn.Parameter(weight.data)
        self.weight.requires_grad = not freeze

        skills_weight_A = weight.new_empty((n_skills, r * self.in_features))
        skills_weight_B = weight.new_empty((n_skills, self.out_features * r))
        self.skills_weight_A = nn.Parameter(skills_weight_A)
        self.skills_weight_B = nn.Parameter(skills_weight_B)
        self.scaling = 1 / self.r

        if bias is not None:
            self.bias = nn.Parameter(bias.data)
            self.bias.requires_grad = not freeze
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            self.skills_weight_A.uniform_(-std, std)
        torch.nn.init.zeros_(self.skills_weight_B)

    def forward(self, input: Tensor) -> Tensor:
        # Provisions for inputs repeated for generation
        assert input.size()[0] % self.task_ids.size(0) == 0
        repeats = input.size()[0] // self.task_ids.size(0)
        if repeats > 1:
            self.task_ids = torch.repeat_interleave(self.task_ids, repeats, dim=0)

        skill_logits = self.skill_logits[self.task_ids]
        if self.is_learned:
            if self.training:
                skill_logits = RelaxedBernoulli(temperature=1., logits=skill_logits).rsample()
            else:
                skill_logits = torch.sigmoid(skill_logits)
        skill_logits = skill_logits / (skill_logits.sum(dim=-1, keepdim=True) + EPS)

        skills_weight_A = torch.mm(skill_logits, self.skills_weight_A).view(input.size()[0], self.in_features, self.r)
        skills_weight_B = torch.mm(skill_logits, self.skills_weight_B).view(input.size()[0], self.r, self.out_features)
        output = torch.matmul(input, skills_weight_A) # bsi,bir->bsr
        output = torch.matmul(output, skills_weight_B) # bsr,bro->bso
        output = F.linear(input, self.weight, self.bias) + output * self.scaling

        return output


class SkilledLTSFTLinear(SkilledModule):
    """ Applies a linear function parameterised by a base bias
    and a weighted average of base and skill weights
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self,
                 n_tasks: int,
                 n_skills: int,
                 skills: Optional[Tensor],
                 weight: Tensor,
                 bias: Optional[Tensor],
                 sparsity: float = 0.1,
                 freeze: bool = True
                 ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape

        if skills is None:
            self.skill_logits = nn.Parameter(torch.empty((n_tasks, n_skills)).uniform_(-1e-3, 1e-3))
            self.is_learned = True
        else:
            self.register_buffer("skill_logits", skills)
            self.is_learned = False

        self.weight = nn.Parameter(weight.data)
        self.weight.requires_grad = not freeze

        indices = itertools.product(range(self.out_features * self.in_features), range(n_skills))
        k = int(self.out_features * self.in_features * n_skills * sparsity)
        indices = random.sample(list(indices), k=k)
        indices = torch.LongTensor(indices).T
        values = torch.zeros((k, ))
        skills_weight = torch.sparse_coo_tensor(indices, values, (self.out_features * self.in_features, n_skills))
        self.skills_weight = nn.Parameter(skills_weight.coalesce())

        if bias is not None:
            self.bias = nn.Parameter(bias.data)
            self.bias.requires_grad = not freeze
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        # Provisions for inputs repeated for generation
        assert input.size()[0] % self.task_ids.size(0) == 0
        repeats = input.size()[0] // self.task_ids.size(0)
        if repeats > 1:
            self.task_ids = torch.repeat_interleave(self.task_ids, repeats, dim=0)

        skill_logits = self.skill_logits[self.task_ids]
        if self.is_learned:
            if self.training:
                skill_logits = RelaxedBernoulli(temperature=1., logits=skill_logits).rsample()
            else:
                skill_logits = torch.sigmoid(skill_logits)
        skill_logits = skill_logits / (skill_logits.sum(dim=-1, keepdim=True) + EPS)

        skills_weight = torch.sparse.mm(self.skills_weight, skill_logits.T).T.view(input.size()[0], self.in_features, self.out_features)
        output = torch.matmul(input, skills_weight) # bsi,bio->bso
        output = F.linear(input, self.weight, self.bias) + output

        # TODO: densify at the end
        # skills_weight = torch.smm(self.skills_weight, skill_logits.T)
        # skills_weight = torch.transpose(skills_weight, 1, 0)

        return output
