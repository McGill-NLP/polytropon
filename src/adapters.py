import math
from typing import Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.init import calculate_gain
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


EPS = 1e-12


class SkilledBitFitLinear(nn.Module):
    """ Applies a linear function parameterised by a base weight
    and a weighted average of base and skill biases
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, n_skills: int, weight: Tensor, bias: Optional[Tensor]) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape

        self.weight = nn.Parameter(weight.data)

        if bias is not None:
            bias = bias.data
            self.bias = nn.Parameter(bias)
            skills_bias = bias.new_empty((n_skills, self.out_features))
            self.skills_bias = nn.Parameter(skills_bias)
        else:
            self.register_parameter('bias', None)
            self.register_parameter('skills_bias', None)

        self.skills = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.skills_bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.skills_bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # Provisions for inputs repeated for generation
        assert input.size()[0] % self.skills.size(0) == 0
        repeats = input.size()[0] // self.skills.size(0)
        if repeats > 1:
            self.skills = torch.repeat_interleave(self.skills, repeats, dim=0)

        bias = torch.mm(self.skills, self.skills_bias) + self.bias.unsqueeze(0)
        return F.linear(input, self.weight) + bias.unsqueeze(1)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class HyperLoRALinear(nn.Module):
    """ Applies a linear function parameterised by a base bias
    and a weighted average of base and task-conditioned weights
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self,
                 n_skills: int,
                 weight: Tensor,
                 bias: Optional[Tensor],
                 n_tasks: int,
                 r: int = 16,
                 freeze: bool = True
                 ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.r = r
        self.finegrained = bool(n_tasks)
        assert self.finegrained

        if self.finegrained:
            self.task_embs = nn.Embedding(n_tasks, n_skills)
            self.task_proj = nn.Sequential(
                nn.Linear(n_skills, n_skills),
                nn.ReLU(),
                nn.Linear(n_skills, n_skills),
            )

        self.weight = nn.Parameter(weight.data)
        if freeze:
            self.weight.requires_grad = False

        self.hyper_weight_A = nn.Linear(n_skills, r * self.in_features, bias=False)
        self.hyper_weight_B = nn.Linear(n_skills, self.out_features * r, bias=False)
        self.scaling = 1 / self.r

        if bias is not None:
            self.bias = nn.Parameter(bias.data)
            if freeze:
                self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)

        self.skills = None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.zeros_(self.hyper_weight_B.weight)

    def forward(self, input: Tensor) -> Tensor:
        # Provisions for inputs repeated for generation
        assert input.size()[0] % self.skills.size(0) == 0
        repeats = input.size()[0] // self.skills.size(0)
        if repeats > 1:
            self.skills = torch.repeat_interleave(self.skills, repeats, dim=0)

        task_embs = self.task_embs(self.skills)
        task_embs = self.task_proj(task_embs)

        hyper_weight_A = self.hyper_weight_A(task_embs).view(input.size()[0], self.in_features, self.r)
        hyper_weight_B = self.hyper_weight_B(task_embs).view(input.size()[0], self.r, self.out_features)
        output = torch.matmul(input, hyper_weight_A) # bsi,bir->bsr
        output = torch.matmul(output, hyper_weight_B) # bsr,bro->bso
        output = F.linear(input, self.weight, self.bias) + output * self.scaling

        return output


class SkilledLoRALinear(nn.Module):
    """ Applies a linear function parameterised by a base bias
    and a weighted average of base and skill weights
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self,
                 n_skills: int,
                 weight: Tensor,
                 bias: Optional[Tensor],
                 n_tasks: int,
                 r: int = 16,
                 freeze: bool = True
                 ) -> None:
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.r = r
        self.finegrained = bool(n_tasks)

        if self.finegrained:
            self.skill_logits = nn.Parameter(torch.empty((n_tasks, n_skills)).uniform_(-1e-3, 1e-3))

        self.weight = nn.Parameter(weight.data)
        if freeze:
            self.weight.requires_grad = False

        skills_weight_A = weight.new_empty((n_skills, r * self.in_features))
        skills_weight_B = weight.new_empty((n_skills, self.out_features * r))
        self.skills_weight_A = nn.Parameter(skills_weight_A)
        self.skills_weight_B = nn.Parameter(skills_weight_B)
        self.scaling = 1 / self.r

        if bias is not None:
            self.bias = nn.Parameter(bias.data)
            if freeze:
                self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)

        self.skills = None

        self.reset_parameters()

    def reset_parameters(self):
        gain = calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
        std = gain / math.sqrt(self.in_features)
        with torch.no_grad():
            self.skills_weight_A.uniform_(-std, std)
        torch.nn.init.zeros_(self.skills_weight_B)

    def forward(self, input: Tensor) -> Tensor:
        if self.finegrained and self.skills.dtype != torch.float32:
            skill_logits = self.skill_logits[self.skills]
            if self.training:
                skill_logits = RelaxedBernoulli(temperature=1., logits=skill_logits).rsample()
            else:
                skill_logits = torch.sigmoid(skill_logits)
            self.skills = skill_logits / (skill_logits.sum(dim=-1, keepdim=True) + EPS)

        # Provisions for inputs repeated for generation
        assert input.size()[0] % self.skills.size(0) == 0
        repeats = input.size()[0] // self.skills.size(0)
        if repeats > 1:
            self.skills = torch.repeat_interleave(self.skills, repeats, dim=0)

        skills_weight_A = torch.mm(self.skills, self.skills_weight_A).view(input.size()[0], self.in_features, self.r)
        skills_weight_B = torch.mm(self.skills, self.skills_weight_B).view(input.size()[0], self.r, self.out_features)
        output = torch.matmul(input, skills_weight_A) # bsi,bir->bsr
        output = torch.matmul(output, skills_weight_B) # bsr,bro->bso
        output = F.linear(input, self.weight, self.bias) + output * self.scaling

        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
