import os
import time
from accelerate import Accelerator
from .util import AverageMeter, ProgressMeter, get_lr

import torch
import torch.nn as nn


def train_robin_model(train_loader, model, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    learning = AverageMeter('Learning Rate', ':1.7f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning, losses],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_training_' + args.model_name + '.csv'))

    accelerator = Accelerator()
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    l1_criterion = nn.L1Loss()
    model.train()
    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        batch = tuple(batch[t].cuda() for t in batch) if torch.cuda.is_available() else tuple(batch[t] for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4]
        }
        outputs = model(**inputs)
        loss = outputs['loss']

        if args.l1_regularization > 0:
            for param in model.parameters():
                loss += args.l1_regularization * l1_criterion(param, torch.zeros_like(param))

        if args.gradient_accumulation_steps > 1:
            loss /= args.gradient_accumulation_steps

        accelerator.backward(loss)
        losses.update(loss.item(), batch[0].size(0))
        learning.update(get_lr(optimizer), batch[0].size(0))

        if ((idx + 1) % args.gradient_accumulation_steps == 0) or ((idx + 1) == len(train_loader)):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        progress.log_metrics(idx)
        if idx % args.print_freq == 0:
            progress.display(idx)

    return losses.avg,


def validate_robin_model(val_loader, model, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_validation_' + args.model_name + '.csv'))

    model.eval()
    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(val_loader):
            data_time.update(time.time() - end)
            batch = tuple(batch[t].cuda() for t in batch) if torch.cuda.is_available() else tuple(
                batch[t] for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4]
            }
            outputs = model(**inputs)
            loss = outputs['loss']

            losses.update(loss.item(), batch[0].size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            progress.log_metrics(idx)
            if idx % args.print_freq == 0:
                progress.display(idx)

        return losses.avg,


def test_robin_model(val_loader, model, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_validation_' + args.model_name + '.csv'))

    model.eval()
    with torch.no_grad():
        end = time.time()
        start_logits, end_logits, logits = [], [], []
        for idx, batch in enumerate(val_loader):
            batch = tuple(batch[t].cuda() for t in batch) if torch.cuda.is_available() else tuple(batch[t] for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1]
            }
            outputs = model(**inputs)
            start_logits.extend(outputs['start_logits'].cpu().numpy())
            end_logits.extend(outputs['end_logits'].cpu().numpy())
            logits.extend(outputs['logits'].cpu().numpy())
            batch_time.update(time.time() - end)
            end = time.time()
            progress.log_metrics(idx)
            if idx % args.print_freq == 0:
                progress.display(idx)

    return start_logits, end_logits, logits
