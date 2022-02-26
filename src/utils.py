def replace_layers(model, old, new_linear, n_skills, n_tasks):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, old, new_linear, n_skills, n_tasks)

        if isinstance(module, old) and name in ["k_proj", "v_proj", "q_proj", "out_proj", "k", "v", "q", "o"]:
            new = new_linear(n_skills, module.weight, module.bias, n_tasks)
            setattr(model, name, new)


def inform_layers(model, old, skills):
    for module in model.children():
        if len(list(module.children())) > 0:
            inform_layers(module, old, skills)

        if isinstance(module, old):
            setattr(module, "skills", skills)
