import os
import time
from accelerate import Accelerator
from .util import AverageMeter, ProgressMeter, get_lr
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import torch
import torch.nn as nn


def train_linear_model(train_loader, model, optimizer, scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    learning = AverageMeter('Learning Rate', ':1.7f')
    f1 = AverageMeter('Macro F1', ':1.5f')
    prec = AverageMeter('Macro Precision', ':1.5f')
    rec = AverageMeter('Macro Recall', ':1.5f')
    acc = AverageMeter('Accuracy', ':1.5f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning, losses, f1, prec, rec, acc],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_training_' + args.model_name + '.csv'))

    accelerator = Accelerator()
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)

    l1_criterion = nn.L1Loss()
    model.train()
    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        # batch = tuple(batch[t].cuda() for t in batch) if torch.cuda.is_available() else tuple(batch[t] for t in batch)
        inputs = {
            "input_ids": batch['input_ids'],
            "attention_mask": batch['attention_mask'],
            "labels": batch['label']
        }
        outputs = model(**inputs)
        loss = outputs[0]

        _labels_ = batch['label'].cpu().numpy()
        _predictions_ = outputs[1].argmax(1).cpu().numpy()

        _f1_score_ = f1_score(_labels_, _predictions_, average='macro', zero_division=0)
        _precision_score_ = precision_score(_labels_, _predictions_, average='macro', zero_division=0)
        _recall_score_ = recall_score(_labels_, _predictions_, average='macro', zero_division=0)
        _accuracy_score_ = accuracy_score(_labels_, _predictions_)

        f1.update(_f1_score_, batch['input_ids'].size(0))
        prec.update(_precision_score_, batch['input_ids'].size(0))
        rec.update(_recall_score_, batch['input_ids'].size(0))
        acc.update(_accuracy_score_, batch['input_ids'].size(0))

        # L1 regularization
        if args.l1_regularization > 0:
            for param in model.parameters():
                loss += args.l1_regularization * l1_criterion(param, torch.zeros_like(param))

        # Gradient Accumulation Step
        if args.gradient_accumulation_steps > 1:
            loss /= args.gradient_accumulation_steps

        accelerator.backward(loss)
        losses.update(loss.item(), batch['input_ids'].size(0))
        learning.update(get_lr(optimizer), batch['input_ids'].size(0))

        if ((idx + 1) % args.gradient_accumulation_steps == 0) or ((idx + 1) == len(train_loader)):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        progress.log_metrics(idx)
        if idx % args.print_freq == 0:
            progress.display(idx)

    return losses.avg


def validate_linear_model(val_loader, model, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    f1 = AverageMeter('Macro F1', ':1.5f')
    prec = AverageMeter('Macro Precision', ':1.5f')
    rec = AverageMeter('Macro Recall', ':1.5f')
    acc = AverageMeter('Accuracy', ':1.5f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, f1, prec, rec, acc],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_validation_' + args.model_name + '.csv'))

    model.eval()
    loss_fct = nn.CrossEntropyLoss()
    num_classes = model.config.num_classes
    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(val_loader):
            data_time.update(time.time() - end)
            inputs = {
                "input_ids": batch['input_ids'].cuda(),
                "attention_mask": batch['attention_mask'].cuda()
            }
            outputs = model(**inputs)
            predicted_labels = outputs[1].view(-1, num_classes)
            true_labels = batch['label'].view(-1)
            loss = loss_fct(predicted_labels, true_labels.cuda())

            _labels_ = true_labels.cpu().numpy()
            _predictions_ = predicted_labels.argmax(1).cpu().numpy()

            _f1_score_ = f1_score(_labels_, _predictions_, average='macro', zero_division=0)
            _precision_score = precision_score(_labels_, _predictions_, average='macro', zero_division=0)
            _recall_score = recall_score(_labels_, _predictions_, average='macro', zero_division=0)
            _accuracy_score = accuracy_score(_labels_, _predictions_)

            f1.update(_f1_score_, batch['input_ids'].size(0))
            prec.update(_precision_score, batch['input_ids'].size(0))
            rec.update(_recall_score, batch['input_ids'].size(0))
            acc.update(_accuracy_score, batch['input_ids'].size(0))

            losses.update(loss.item(), batch['input_ids'].size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            progress.log_metrics(idx)
            if idx % args.print_freq == 0:
                progress.display(idx)

        return losses.avg


def test_linear_model(val_loader, model, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_validation_' + args.model_name + '.csv'))

    model.eval()
    num_classes = model.config.num_classes
    with torch.no_grad():
        end = time.time()
        _ids_ = []
        _predictions_ = []
        _true_labels_ = []
        logits = []
        for idx, batch in enumerate(val_loader):
            inputs = {
                "input_ids": batch['input_ids'].cuda(),
                "attention_mask": batch['attention_mask'].cuda()
            }
            outputs = model(**inputs)
            predicted_labels = outputs[1].view(-1, num_classes)
            true_labels = batch['label'].view(-1).cuda()
            predictions = predicted_labels.argmax(1)
            _ids_.extend(batch['id'])
            _predictions_.extend(predictions.cpu().numpy().astype(str))
            _true_labels_.extend(true_labels.cpu().numpy().astype(str))
            probabilities = nn.functional.softmax(predicted_labels, dim=1)
            logits.extend(probabilities[:, 1].cpu().numpy().astype(float))

            batch_time.update(time.time() - end)
            end = time.time()
            progress.log_metrics(idx)
            if idx % args.print_freq == 0:
                progress.display(idx)

    return logits, _ids_, _predictions_, _true_labels_
