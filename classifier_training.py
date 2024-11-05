import sys
import os
import time
import warnings
import json
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets import load_dataset
from models.linear_classifier import LinearClassifier, LinearConfig
from models.mlp_classifier import MLPClassifier, MLPConfig
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from utils import load_state, save_model_checkpoint
from pipeline.training_classifier_pipeline import train_linear_model, validate_linear_model, test_linear_model

from data_models.data_classification import ClassificationDataset


def get_args():

    parser = argparse.ArgumentParser(description='PyTorch RoBIn Training')

    # parameters
    parser.add_argument("--model_name", default='distilbert-base-uncased', type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--data_folder", default='./data', type=str,
                        help="Path to data folder.")
    parser.add_argument("--dataset_name", default='robqa', type=str,
                        help="Name of the dataset to use.")
    parser.add_argument("--answers_path", default=None, type=str,
                        help="Path to answers file to be evaluated.")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="Number of classes for classification Default: 2 (LOW - 1 or HIGH/UNCLEAR RISK OF BIAS - 0).")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this "
                             "will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--dropout", default=0.2, type=float,
                        help="Dropout probability.")
    parser.add_argument("--l1_regularization", default=0.0, type=float,
                        help="L1 regularization coefficient.")
    parser.add_argument("--warmup_steps", default=1000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--pretrained", default=False, action='store_true',
                        help="Use pretrained model.")
    parser.add_argument("--pos_weight", default=1.0, type=float,
                        help="Weight for positive class in BCE loss.")
    parser.add_argument("--loss_fn_cls", default='bce', type=str,
                        help="Loss function for classification.")
    parser.add_argument("--mlp", default=False, action='store_true',
                        help="Use MLP classifier.")

    # distributed training parameters
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed value (default: None)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Number of workers for the data loader.")

    parser.add_argument("--evaluate", action='store_true',
                        help="Evaluate model on test set.")
    parser.add_argument("--last_epoch", action='store_true',
                        help="Evaluate last epoch model.")

    parser.add_argument("--print_freq", default=10, type=int,
                        help="Print frequency.")

    args = parser.parse_args()
    args.model_path = "./save/{}_models".format(args.model_name)

    args.model_filename = 'linear_{}_len_{}_bsz_{}_lr_{}_wd_{}'.format(args.dataset_name,
                                                                             args.max_seq_length,
                                                                             args.batch_size,
                                                                             args.learning_rate,
                                                                             args.weight_decay)

    if args.mlp:
        args.model_filename += '_mlp'

    args.model_filename += '_' + args.loss_fn_cls

    args.start_epoch = 0
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.save_folder = os.path.join(args.model_path, args.model_filename)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
        os.makedirs(os.path.join(args.save_folder, 'last_epoch'), exist_ok=True)

    return args


def trainer_fn(args, model, tokenizer):
    # treinar modelo linear do robin classifier apenas
    # carregar dataset
    best_loss = None
    val_loss = None

    dataset = load_dataset('json', data_files={
        'train': os.path.join(args.data_folder, 'rob_dataset_train.json'),
        'validation': os.path.join(args.data_folder, 'rob_dataset_test.json')})

    class_dict = {'UNCLEAR': 0, 'HIGH': 0, 'LOW': 1}

    train_sentences = [data['answers']['text'][0] for _, data in enumerate(dataset['train'])]
    train_ids = [data['id'] for _, data in enumerate(dataset['train'])]
    train_labels = [class_dict[data['label']] for _, data in enumerate(dataset['train'])]

    val_sentences = [data['answers']['text'][0] for _, data in enumerate(dataset['validation'])]
    val_ids = [data['id'] for _, data in enumerate(dataset['validation'])]
    val_labels = [class_dict[data['label']] for _, data in enumerate(dataset['validation'])]

    train_dataset = ClassificationDataset(train_ids, train_sentences, train_labels, tokenizer, max_length=args.max_seq_length)
    val_dataset = ClassificationDataset(val_ids, val_sentences, val_labels, tokenizer, max_length=args.max_seq_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    total_steps = len(train_loader) * args.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)
    for epoch in range(args.start_epoch, args.epochs):
        try:
            time1 = time.time()
            train_loss = train_linear_model(train_loader, model, optimizer, scheduler, epoch, args)
            time2 = time.time()

            print('Epoch {}, total time {:.2f}, training loss {:.3f}'.format(epoch, time2 - time1, train_loss))

            val_time1 = time.time()
            val_loss = validate_linear_model(val_loader, model, epoch, args)
            val_time2 = time.time()

            print('Epoch {}, total time {:.2f}, validation loss {:.3f}'.format(epoch, val_time2 - val_time1, val_loss))

            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                print("New best loss: {:.3f}".format(best_loss))
                save_model_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'val_loss': val_loss,
                    'optimizer': optimizer.state_dict(),
                }, args.save_folder)
                model.save_pretrained(os.path.join(args.save_folder, 'pretrained'))
                tokenizer.save_pretrained(os.path.join(args.save_folder, 'pretrained'))

            # save last epoch
            save_model_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'val_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.save_folder, 'last_epoch'))
            model.save_pretrained(os.path.join(args.save_folder, 'last_epoch', 'pretrained'))
            tokenizer.save_pretrained(os.path.join(args.save_folder, 'last_epoch', 'pretrained'))

            print("Best loss: {:.3f}".format(best_loss))
        except Exception as e:
            print('Error: ', e)
            continue

    print("Done!")


def test_fn(args, model, tokenizer):
    # testar modelo linear do robin classifier apenas
    # carregar dataset
    dataset = load_dataset('json', data_files={
        'test': os.path.join(args.data_folder, 'rob_dataset_test.json')})

    class_dict = {'UNCLEAR': 0, 'HIGH': 0, 'LOW': 1}

    test_ids = [data['id'] for _, data in enumerate(dataset['test'])]
    test_labels = [class_dict[data['label']] for _, data in enumerate(dataset['test'])]

    if args.answers_path is not None:
        with open(args.answers_path, 'r', encoding='utf-8') as f:
            answers = json.load(f)
        answers = {data['id']: data['prediction_text'] for data in answers}
        test_sentences = []
        for _id_ in test_ids:
            _new_answer_ = answers[_id_].split('.')
            if len(_new_answer_) > 1 and len(_new_answer_[1].split()) == 1:
                test_sentences.append(_new_answer_[0] + '.')
            else:
                test_sentences.append(answers[_id_])
    else:
        test_sentences = [data['answers']['text'][0] for _, data in enumerate(dataset['test'])]

    test_dataset = ClassificationDataset(test_ids, test_sentences, test_labels, tokenizer, max_length=args.max_seq_length)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )

    test_probs, _ids_, _preds_, _true_labels_ = test_linear_model(test_loader, model, 0, args)
    content = {
        'ids': _ids_,
        'preds': _preds_,
        'true_labels': _true_labels_,
        'probs': test_probs
    }

    if args.last_epoch:
        with open(os.path.join(args.save_folder, 'last_epoch', 'test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(content, f)
    else:
        with open(os.path.join(args.save_folder, 'test_results.json'), 'w', encoding='utf-8') as f:
            json.dump(content, f)

    print("Done!")


if __name__ == "__main__":

    __args = get_args()

    if __args.seed is not None:
        torch.manual_seed(__args.seed)
        torch.cuda.manual_seed_all(__args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if __args.gpu is not None:
        device = torch.device(f'cuda:{__args.gpu}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## MODEL DEFINITION
    if __args.pretrained:
        if __args.last_epoch:
            pretrained_path = os.path.join(__args.save_folder, 'last_epoch', 'pretrained')
        else:
            pretrained_path = os.path.join(__args.save_folder, 'pretrained')

        __tokenizer__ = AutoTokenizer.from_pretrained(pretrained_path)
        if __args.mlp:
            __model__ = MLPClassifier.from_pretrained(pretrained_path)
        else:
            __model__ = LinearClassifier.from_pretrained(pretrained_path)
    else:
        __tokenizer__ = AutoTokenizer.from_pretrained(__args.model_name, clean_up_tokenization_spaces=True)

        if __args.mlp:
            __config__ = MLPConfig(model_name=__args.model_name, num_classes=__args.num_classes, dropout=__args.dropout,
                                   pos_weight=__args.pos_weight, loss_fn_cls=__args.loss_fn_cls)
            __model__ = MLPClassifier(__config__)
        else:
            __config__ = LinearConfig(model_name=__args.model_name, num_classes=__args.num_classes, dropout=__args.dropout,
                                      pos_weight=__args.pos_weight, loss_fn_cls=__args.loss_fn_cls)
            __model__ = LinearClassifier(__config__)

    __model__.to(device)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        __model__ = __model__.cuda()

    if __args.evaluate:
        print('Evaluating model...')
        ## EVALUATE MODEL
        test_fn(__args, __model__, __tokenizer__)
    else:
        print('Training model...')
        ## TRAIN MODEL
        trainer_fn(__args, __model__, __tokenizer__)


    print('Done!')
    sys.exit(0)
