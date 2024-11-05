import os
import time
from pipeline.util import AverageMeter, ProgressMeter, get_lr
from transformers import BartTokenizer
from pipeline.util import create_metrics_object, shift_tokens_right
from textstat import flesch_kincaid_grade
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pipeline.metrics.sari import SARIsent
from accelerate import Accelerator
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from models.gen_robin import GenerativeRoBInClassifier
import evaluate
sari = evaluate.load('sari')
bleu = evaluate.load('bleu')
THRESHOLD = 0.5

def train(dataloader, epoch, model: GenerativeRoBInClassifier, tokenizer: BartTokenizer, optimizer, criterion, scheduler, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    perplexity = AverageMeter('Perplexity', ':1.5f')
    learning = AverageMeter('Learning', ':1.5f')
    bleu_score = AverageMeter('BLEU', ':1.5f')
    sari_score = AverageMeter('SARI', ':1.5f')
    fkgl_score = AverageMeter('FKGL', ':1.5f')
    accuracy_score_ = AverageMeter('Accuracy', ':1.5f')
    precision_score_ = AverageMeter('Precision', ':1.5f')
    recall_score_ = AverageMeter('Recall', ':1.5f')
    f1_score_ = AverageMeter('F1', ':1.5f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, learning, losses, perplexity, bleu_score, sari_score, fkgl_score, accuracy_score_, precision_score_, recall_score_, f1_score_],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_training_' + args.model_name + '.csv'))

    res = create_metrics_object()

    accelerator = Accelerator()
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    model.train()
    total_loss = 0
    total_perplexity = 0

    end = time.time()
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        data_time.update(time.time() - end)
        bsz = batch[0].size(0)
        batch = tuple(t.cuda() for t in batch) if torch.cuda.is_available() else batch
        decoded_input_ids = shift_tokens_right(batch[2], tokenizer.pad_token_id)
        input = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'decoder_input_ids': decoded_input_ids,
            'labels': batch[4]
        }

        output = model(**input, use_cache=False)
        logits = output[0]['logits']
        prediction = torch.argmax(logits, dim=2)
        classifier_logits = output[1]['classifier_logits']
        classifier_predictions = (torch.sigmoid(classifier_logits) > THRESHOLD).clone().detach().cpu()
        true_labels = batch[4].clone().detach().cpu()

        accuracy_score_.update(accuracy_score(true_labels, classifier_predictions), bsz)
        precision_score_.update(precision_score(true_labels, classifier_predictions, zero_division=True, average='macro'), bsz)
        recall_score_.update(recall_score(true_labels, classifier_predictions, zero_division=True, average='macro'), bsz)
        f1_score_.update(f1_score(true_labels, classifier_predictions, zero_division=True, average='macro'), bsz)

        batch_source = tokenizer.batch_decode(input['input_ids'], skip_special_tokens=True)
        batch_reference = tokenizer.batch_decode(batch[2], skip_special_tokens=True)
        batch_prediction = tokenizer.batch_decode(prediction, skip_special_tokens=True)
        res['source'] += batch_source
        res['target'] += batch_reference
        res['output'] += batch_prediction

        _bleu_score_ = bleu.compute(predictions=batch_prediction, references=batch_reference)
        bleu_score.update(_bleu_score_['bleu'], bsz)
        [res['bleu'].append(_bleu_score_['bleu']) for _ in range(bsz)]

        _total_sari_score_ = 0.
        _total_fkgl_score_ = 0.
        for source, prediction, reference in zip(batch_source, batch_prediction, batch_reference):
            _sari_score_ = SARIsent(source, prediction, [reference])
            _fkgl_score_ = flesch_kincaid_grade(prediction)
            _total_sari_score_ += _sari_score_
            _total_fkgl_score_ += _fkgl_score_
            res['sari'].append(_sari_score_)
            res['fkgl'].append(_fkgl_score_)

        sari_score.update(_total_sari_score_/bsz, bsz)
        fkgl_score.update(_total_fkgl_score_/bsz, bsz)


        loss = criterion(logits.view(-1, logits.size(-1)), batch[2].view(-1))
        classifier_loss = nn.BCEWithLogitsLoss()(classifier_logits.view(-1), batch[4].float().view(-1))
        loss += classifier_loss

        if args.gradient_accumulation_steps > 1:
            loss /= args.gradient_accumulation_steps

        accelerator.backward(loss)
        # compute bleu score


        learning.update(get_lr(optimizer), bsz)

        if ((idx + 1) % args.gradient_accumulation_steps == 0) or ((idx + 1) == len(dataloader)):
            optimizer.step()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scheduler.step()

        losses.update(loss.item(), bsz)
        _perplexity_ = torch.exp(loss)
        [res['perplexity'].append(_perplexity_.item()) for _ in range(bsz)]
        perplexity.update(_perplexity_.item(), bsz)
        total_loss += loss.item()
        total_perplexity += torch.exp(loss)

        batch_time.update(time.time() - end)
        end = time.time()

        progress.log_metrics(idx)
        if (idx + 1) % args.print_freq == 0:
            progress.display(idx)

    return total_loss / len(dataloader), total_perplexity / len(dataloader), res


def validate(dataloader, epoch, model, tokenizer, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':1.5f')
    perplexity = AverageMeter('Perplexity', ':1.5f')
    bleu_score = AverageMeter('BLEU', ':1.5f')
    sari_score = AverageMeter('SARI', ':1.5f')
    fkgl_score = AverageMeter('FKGL', ':1.5f')
    accuracy_score_ = AverageMeter('Accuracy', ':1.5f')
    precision_score_ = AverageMeter('Precision', ':1.5f')
    recall_score_ = AverageMeter('Recall', ':1.5f')
    f1_score_ = AverageMeter('F1', ':1.5f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, losses, perplexity, bleu_score, sari_score, fkgl_score, accuracy_score_, precision_score_, recall_score_, f1_score_],
        prefix="Epoch: [{}]".format(epoch),
        logfile=os.path.join(args.save_folder, 'log_validation_' + args.model_name + '.csv'))

    res = create_metrics_object()

    model.eval()
    total_loss = 0
    total_perplexity = 0

    with torch.no_grad():
        end = time.time()
        for idx, batch in enumerate(dataloader):
            bsz = batch[0].size(0)
            batch = tuple(t.cuda() for t in batch) if torch.cuda.is_available() else batch
            input = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'decoder_input_ids': shift_tokens_right(batch[2], tokenizer.pad_token_id),
                'labels': batch[4]
            }

            output = model(**input, use_cache=False)
            logits = output[0]['logits']
            prediction = torch.argmax(logits, dim=2)
            classifier_logits = output[1]['classifier_logits']
            classifier_predictions = (torch.sigmoid(classifier_logits) > THRESHOLD).clone().detach().cpu()
            true_labels = batch[4].clone().detach().cpu()

            accuracy_score_.update(accuracy_score(true_labels, classifier_predictions), bsz)
            precision_score_.update(precision_score(true_labels, classifier_predictions, zero_division=True, average='macro'), bsz)
            recall_score_.update(recall_score(true_labels, classifier_predictions, zero_division=True, average='macro'), bsz)
            f1_score_.update(f1_score(true_labels, classifier_predictions, zero_division=True, average='macro'), bsz)

            batch_source = tokenizer.batch_decode(input['input_ids'], skip_special_tokens=True)
            batch_reference = tokenizer.batch_decode(batch[2], skip_special_tokens=True)
            batch_prediction = tokenizer.batch_decode(prediction, skip_special_tokens=True)
            res['source'] += batch_source
            res['target'] += batch_reference
            res['output'] += batch_prediction

            _bleu_score_ = bleu.compute(predictions=batch_prediction, references=batch_reference)
            bleu_score.update(_bleu_score_['bleu'], bsz)
            [res['bleu'].append(_bleu_score_['bleu']) for l in range(bsz)]

            _total_sari_score_ = 0.
            _total_fkgl_score_ = 0.
            for source, prediction, reference in zip(batch_source, batch_prediction, batch_reference):
                _sari_score_ = SARIsent(source, prediction, [reference])
                _fkgl_score_ = flesch_kincaid_grade(prediction)
                _total_sari_score_ += _sari_score_
                _total_fkgl_score_ += _fkgl_score_
                res['sari'].append(_sari_score_)
                res['fkgl'].append(_fkgl_score_)

            sari_score.update(_total_sari_score_ / bsz, bsz)
            fkgl_score.update(_total_fkgl_score_ / bsz, bsz)

            loss = criterion(logits.view(-1, logits.size(-1)), batch[2].view(-1))
            classifier_loss = nn.BCEWithLogitsLoss()(classifier_logits.view(-1), batch[4].float().view(-1))
            loss += classifier_loss

            total_loss += loss.item()

            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps

            total_perplexity += torch.exp(loss)
            losses.update(loss.item(), bsz)

            _perplexity_ = torch.exp(loss)
            [res['perplexity'].append(_perplexity_.item()) for _ in range(bsz)]
            perplexity.update(_perplexity_.item(), bsz)
            batch_time.update(time.time() - end)
            end = time.time()

            progress.log_metrics(idx)
            if (idx + 1) % args.print_freq == 0:
                progress.display(idx)

    return total_loss / len(dataloader), total_perplexity / len(dataloader), res
