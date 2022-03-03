from torch import nn


ATTENTION_LINEARS = ["k_proj", "v_proj", "q_proj", "out_proj", "k", "v", "q", "o"]


def replace_layers(model, adapter_class, n_tasks, n_skills, skills, only_attention=True):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, adapter_class, n_tasks, n_skills, skills, only_attention=only_attention)

        if isinstance(module, nn.Linear) and (name in ATTENTION_LINEARS or not only_attention):
            new_linear = adapter_class(n_tasks, n_skills, skills, module.weight, module.bias)
            setattr(model, name, new_linear)


def inform_layers(model, adapter_class, value):
    for module in model.children():
        if len(list(module.children())) > 0:
            inform_layers(module, adapter_class, value)

        if isinstance(module, adapter_class):
            module.task_ids = value
