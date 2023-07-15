from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(opt, optimizer, n_iter_per_epoch):
    num_steps = int(opt.max_epochs * n_iter_per_epoch)
    warmup_steps = int(opt.warmup_epochs * n_iter_per_epoch)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        # 第一次重新启动经历的迭代次数
        # t_initial=(num_steps - warmup_steps) if opt.lr_scheduler_warmup_prefix else num_steps,
        t_initial=int((num_steps+1) * opt.t_initial_multiplier),
        # 训练期间要使用的最低学习率，学习率永远不会低于这个值
        lr_min=opt.min_lr,
        # 预热期间的初始学习率
        warmup_lr_init=opt.warmup_lr,
        # warmup经历几个迭代次数
        warmup_t=warmup_steps,
        # (重要！)迭代次数是否以epoch而不是batch更新次数给出
        t_in_epochs=False,
        # 预热后的初始学习率是否处于起点位置
        warmup_prefix=opt.lr_scheduler_warmup_prefix,
        cycle_decay=0.5,
        cycle_limit=10)

    return lr_scheduler
