from torch import optim as optim


def build_optimizer(opt, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    parameters = set_weight_decay(model, opt)
    optimizer = optim.AdamW(parameters, eps=opt.optimizer_eps, betas=opt.optimizer_beats,
                            lr=opt.base_lr, weight_decay=opt.weight_decay)
    return optimizer


# /-------- 使用set_weight_decay()将需要权重衰减和不需要权重衰减的参数分开保存获得parameters --------/ #
def set_weight_decay(model, opt):
    has_decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:  # 仅处理需要计算梯度的参数
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]  # 注意
