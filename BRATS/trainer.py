import os
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.distributed
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch


def train_epoch(args, model, loader, epoch, optimizer, scheduler, scaler, loss_func, writer):  # 起始行16
    model.train()
    start_time = time.time()
    num_steps = len(loader)  # 获得一个epoch的迭代步数（几个batchsize）
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            writer.add_scalar(tag="learning_rate/step", scalar_value=optimizer.param_groups[0]['lr'],
                              global_step=epoch*num_steps+idx)
            writer.add_scalar(tag="train_loss/step", scalar_value=run_loss.avg,
                              global_step=epoch*num_steps+idx)
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "\tloss: {:.4f}".format(run_loss.avg),
                "\ttime {:.2f}s".format(time.time() - start_time),
            )
        scheduler.step_update(epoch * num_steps + idx)
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(args, model, loader, acc_func, model_inferer, post_sigmoid, post_pred):
    model.eval()
    run_acc = AverageMeter()
    with torch.no_grad():
        start_time = time.time()
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)
            with autocast(enabled=args.amp):
                logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            #
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                Dice_TC = run_acc.avg[0]
                Dice_WT = run_acc.avg[1]
                Dice_ET = run_acc.avg[2]
                avg_acc = np.mean(run_acc.avg)
                print("Val idx {}/{} mean acc: ".format(idx, len(loader)), avg_acc,
                      "\tDice_TC:", Dice_TC, "\tDice_WT:", Dice_WT, "\tDice_ET:", Dice_ET,
                      "\ttime {:.2f}s".format(time.time() - start_time),
                      )
            start_time = time.time()

    return run_acc.avg


def save_checkpoint(model, epoch, args, best_acc, filename="model_best.pt"):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(args, model, train_loader, val_loader, start_epoch, optimizer, scheduler, loss_func,
                 acc_func, model_inferer, post_sigmoid, post_pred, semantic_classes):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    # ------------------------ 训练阶段 ------------------------ #
    val_acc_max400 = val_acc_max600 = val_acc_max800 = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(args=args, model=model, loader=train_loader, epoch=epoch,
                                 optimizer=optimizer, scheduler=scheduler,
                                 loss_func=loss_func, scaler=scaler, writer=writer)
        if args.rank == 0:
            writer.add_scalar("train_loss/epoch", train_loss, epoch)
            print(
                "Final training epoch {}/{}".format(epoch, args.max_epochs - 1),
                "\tloss: {:.4f}".format(train_loss),
                "\ttime {:.2f}s".format(time.time() - epoch_time),
            )

        # ------------------------ 验证阶段 ------------------------ #
        if (epoch + 1) % args.val_every == 0 or epoch == 0 or epoch + 1 == args.max_epochs:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(args=args, model=model, loader=val_loader,
                                    acc_func=acc_func, model_inferer=model_inferer,
                                    post_sigmoid=post_sigmoid, post_pred=post_pred)

            Mean_Val_Dice = np.mean(val_avg_acc)
            if args.rank == 0:
                writer.add_scalar("Mean_Val_Dice", Mean_Val_Dice, epoch)
                for val_channel_ind in range(len(semantic_classes)):
                    writer.add_scalar("Val_Dice/"+semantic_classes[val_channel_ind], val_avg_acc[val_channel_ind], epoch)
                print("Mean_Val_Dice =", Mean_Val_Dice, "\ttime: ", (time.time() - epoch_time))  # 结束行156

                # ------------------------ 保存阶段 ------------------------ #
                if epoch < 400:
                    if Mean_Val_Dice > val_acc_max400:
                        print("****** new best val_acc_max_0~400 ({:.6f} --> {:.6f}). ******".format(val_acc_max400, Mean_Val_Dice))
                        val_acc_max400 = Mean_Val_Dice
                        if args.rank == 0 and args.save_checkpoint:
                            save_checkpoint(args=args, model=model, epoch=epoch, best_acc=val_acc_max400,
                                            filename="model_best1.pt")
                elif 400 <= epoch < 600:
                    if Mean_Val_Dice > val_acc_max600:
                        print("****** new best val_acc_max_400~600 ({:.6f} --> {:.6f}). ******".format(val_acc_max600, Mean_Val_Dice))
                        print("****** 1. val_acc_max_0~400 = ({:.6f}) ******".format(val_acc_max400))
                        val_acc_max600 = Mean_Val_Dice
                        if args.rank == 0 and args.save_checkpoint:
                            save_checkpoint(args=args, model=model, epoch=epoch, best_acc=val_acc_max600,
                                            filename="model_best2.pt")
                elif 600 <= epoch < 800:
                    if Mean_Val_Dice > val_acc_max800:
                        print("****** new best val_acc_max_600~800 ({:.6f} --> {:.6f}). ******".format(val_acc_max800, Mean_Val_Dice))
                        print("****** 1. val_acc_max_0~400 = ({:.6f}). ******".format(val_acc_max400))
                        print("****** 2. val_acc_max_400~600 = ({:.6f}). ******".format(val_acc_max600))
                        val_acc_max800 = Mean_Val_Dice
                        if args.rank == 0 and args.save_checkpoint:
                            save_checkpoint(args=args, model=model, epoch=epoch, best_acc=val_acc_max800,
                                            filename="model_best3.pt")

    print("Training Finished !, Best Accuracy: ", val_acc_max400, val_acc_max600, val_acc_max800)
