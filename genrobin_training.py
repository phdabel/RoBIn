import sys
import os
import time
import warnings
import json
import argparse
import torch
from torch.optim.adam import Adam
import torch.backends.cudnn as cudnn
from transformers import BartTokenizer, default_data_collator, get_linear_schedule_with_warmup
from datasets import load_dataset
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

from models.gen_robin import GenerativeRoBInClassifier, GenerativeRoBInConfig
from torch.utils.data import DataLoader
from data_models.data_robin import RoBInDataset
from pipeline.training_genrobin import train, validate
from pipeline.testing_genrobin import evaluate_model


def get_args():
    parser = argparse.ArgumentParser(description='Arguments for training and evaluating models')

    # Required parameters
    parser.add_argument("--model_name", default='GanjinZero/biobart-base', type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--data_folder", default='./data', type=str,
                        help="Path to data folder.")
    parser.add_argument("--dataset_name", default='robqa', type=str,
                        help="Name of the dataset to use.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this "
                             "will be truncated, sequences shorter will be padded.")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=1000, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--l1_regularization", default=0.1, type=float,
                        help="L1 regularization coefficient.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--print_freq", default=10, type=int,
                        help="Print frequency.")

    parser.add_argument("--pretrained", default=False, action='store_true',
                        help="Use pretrained model.")
    parser.add_argument("--resume", action='store_true',
                        help="Resume training from checkpoint.")
    parser.add_argument("--last_epoch", default=False, action='store_true',
                        help="Resume training/evaluate from the last epoch.")
    parser.add_argument("--evaluate", action='store_true',
                        help="Evaluate model on test set.")

    # distributed training parameters
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed value (default: None)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of data loading workers (default: 1)')

    args = parser.parse_args()
    args.start_epoch = 0
    # set model path, name, and save folder
    args.model_path = "./save/{}_models".format(args.model_name)
    args.model_filename = '{}_len_{}_bsz_{}_lr_{}'.format(args.dataset_name,
                                                          args.max_seq_length,
                                                          args.batch_size,
                                                          args.learning_rate)
    args.save_folder = os.path.join(args.model_path, args.model_filename)
    if not os.path.exists(args.save_folder):
        print("Creating save folder: {}".format(args.save_folder))
        os.makedirs(args.save_folder)

    return args


def trainer_fn(args, model, tokenizer):

    print('Loading data...')
    dataset = load_dataset('json', data_files={
        'train': os.path.join(args.data_folder, 'rob_dataset_train.json'),
        'validation': os.path.join(args.data_folder, 'rob_dataset_test.json')})


    training_inputs = tokenizer.batch_encode_plus([t_data['question'] + tokenizer.sep_token + t_data['context'] + tokenizer.eos_token for t_data in dataset['train']],
                                                  return_tensors='pt', return_attention_mask=True,
                                                  truncation=True, return_overflowing_tokens=True,
                                                  stride=128,
                                                  padding='max_length', max_length=args.max_seq_length)

    validation_inputs = tokenizer.batch_encode_plus([t_data['question'] + tokenizer.sep_token + t_data['context'] + tokenizer.eos_token for t_data in dataset['validation']],
                                                    return_tensors='pt', return_attention_mask=True,
                                                    truncation=True, return_overflowing_tokens=True,
                                                    stride=128,
                                                    padding='max_length', max_length=args.max_seq_length)


    training_answers = tokenizer.batch_encode_plus([t_data['answers']['text'][0] for t_data in dataset['train']],
                                                  return_tensors='pt', return_attention_mask=True,
                                                  truncation=True, return_overflowing_tokens=True,
                                                  stride=128,
                                                  padding='max_length', max_length=args.max_seq_length)

    validation_answers = tokenizer.batch_encode_plus([t_data['answers']['text'][0] for t_data in dataset['validation']],
                                                    return_tensors='pt', return_attention_mask=True,
                                                    truncation=True, return_overflowing_tokens=True,
                                                    stride=128,
                                                    padding='max_length', max_length=args.max_seq_length)


    class_dict = {'UNCLEAR': 0, 'HIGH': 0, 'LOW': 1}
    training_labels = torch.tensor([class_dict[t_data['label']] for t_data in dataset['train']])
    validation_labels = torch.tensor([class_dict[t_data['label']] for t_data in dataset['validation']])


    train_dataset = TensorDataset(training_inputs['input_ids'], training_inputs['attention_mask'],
                                  training_answers['input_ids'], training_answers['attention_mask'],
                                  training_labels)
    val_dataset = TensorDataset(validation_inputs['input_ids'], validation_inputs['attention_mask'],
                                validation_answers['input_ids'], validation_answers['attention_mask'],
                                validation_labels)

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

    global best_val_metric
    best_val_metric = None

    total_steps = len(train_loader) * args.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)

    for epoch in range(args.start_epoch, args.epochs):
        time1 = time.time()
        train_loss, train_perplexity, train_res = train(train_loader, epoch, model, tokenizer, optimizer, __criterion__,
                                                        scheduler, args)
        time2 = time.time()
        print('Train epoch {}, loss {:.7f}, total time {:.2f}'.format(epoch, train_loss, time2 - time1))
        print('Perplexity: {}'.format(train_perplexity))

        dev_time1 = time.time()
        dev_loss, val_perplexity, dev_res = validate(val_loader, epoch, model, tokenizer, __criterion__, args)
        dev_time2 = time.time()
        print('Dev epoch {}, loss {:.7f}, total time {:.2f}'.format(epoch, dev_loss, dev_time2 - dev_time1))
        print('Perplexity: {}'.format(val_perplexity))

        if best_val_metric is None or dev_loss < best_val_metric:
            print("New best model for val metric: {:.3f}! Saving model...".format(dev_loss))
            best_val_metric = dev_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_val_metric': best_val_metric,
                'optimizer': optimizer.state_dict(),
            }, args.save_folder, 'checkpoint.pth.tar')

            model.save_pretrained(os.path.join(args.save_folder, 'pretrained'))
            tokenizer.save_pretrained(os.path.join(args.save_folder, 'pretrained'))

        model.save_pretrained(os.path.join(args.save_folder, 'last_epoch', 'pretrained'))
        tokenizer.save_pretrained(os.path.join(args.save_folder, 'last_epoch', 'pretrained'))

        print('Saving results...')
        with open(os.path.join(args.save_folder, 'train_res.json'), 'w') as f:
            json.dump(train_res, f)
        with open(os.path.join(args.save_folder, 'val_res.json'), 'w') as f:
            json.dump(dev_res, f)
        train_res_df = pd.DataFrame(train_res)
        train_res_df.to_csv(os.path.join(args.save_folder, 'train_res.csv'), index=False)
        val_res_df = pd.DataFrame(dev_res)
        val_res_df.to_csv(os.path.join(args.save_folder, 'val_res.csv'), index=False)


def save_checkpoint(state, save_folder, filename: str):
    filename = os.path.join(save_folder, filename)
    torch.save(state, filename)


def test_fn(args, model, tokenizer):

    dataset = load_dataset('json', data_files={
        'train': os.path.join(args.data_folder, 'rob_dataset_train.json'),
        'validation': os.path.join(args.data_folder, 'rob_dataset_test.json')})

    validation_inputs = tokenizer.batch_encode_plus(
        [t_data['question'] + tokenizer.sep_token + t_data['context'] + tokenizer.eos_token for t_data in
         dataset['validation']],
        return_tensors='pt', return_attention_mask=True,
        truncation=True, return_overflowing_tokens=True,
        stride=128,
        padding='max_length', max_length=args.max_seq_length)

    validation_answers = tokenizer.batch_encode_plus([t_data['answers']['text'][0] for t_data in dataset['validation']],
                                                    return_tensors='pt', return_attention_mask=True,
                                                    truncation=True, return_overflowing_tokens=True,
                                                    stride=128,
                                                    padding='max_length', max_length=args.max_seq_length)

    class_dict = {'UNCLEAR': 0, 'HIGH': 0, 'LOW': 1}
    validation_labels = torch.tensor([class_dict[t_data['label']] for t_data in dataset['validation']])

    ids = [t_data['id'] for t_data in dataset['validation']]

    val_dataset = TensorDataset(validation_inputs['input_ids'], validation_inputs['attention_mask'],
                                validation_answers['input_ids'], validation_answers['attention_mask'],
                                validation_labels)

    test_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False
    )
    test_res = evaluate_model(ids, test_loader, model, tokenizer, args)

    test_res_df = pd.DataFrame(test_res)
    if args.last_epoch:
        test_res_df.to_csv(os.path.join(args.save_folder, 'last_epoch', 'test_res.csv'), index=False)
    else:
        test_res_df.to_csv(os.path.join(args.save_folder, 'test_res.csv'), index=False)


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

    # set the device
    if __args.gpu is not None:
        device = torch.device(f'cuda:{__args.gpu}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')
    elif __args.gpu == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    if __args.pretrained:
        if __args.last_epoch:
            print('Loading last epoch model...')
            __model__ = GenerativeRoBInClassifier.from_pretrained(os.path.join(__args.save_folder, 'last_epoch', 'pretrained'))
            __tokenizer__ = BartTokenizer.from_pretrained(os.path.join(__args.save_folder, 'last_epoch', 'pretrained'))
        else:
            print('Loading pretrained model...')
            __model__ = GenerativeRoBInClassifier.from_pretrained(os.path.join(__args.save_folder, 'pretrained'))
            __tokenizer__ = BartTokenizer.from_pretrained(os.path.join(__args.save_folder, 'pretrained'))
    else:
        __tokenizer__ = BartTokenizer.from_pretrained(__args.model_name)
        __config__ = GenerativeRoBInConfig(__args.model_name, pos_weight=0.482)
        __config__.eos_token_id = __tokenizer__.eos_token_id
        __model__ = GenerativeRoBInClassifier(__config__)
        __criterion__ = torch.nn.CrossEntropyLoss(ignore_index=__tokenizer__.pad_token_id)

    __model__.to(device)

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