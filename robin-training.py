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
from models.robin_classifier import RoBInClassifier, RoBInConfig
from transformers import AutoTokenizer, default_data_collator, get_cosine_schedule_with_warmup
from datasets import load_dataset
from utils import load_state, save_model_checkpoint
from data_models.data_robin import RoBInDataset
from pipeline.training_robin_pipeline import train_robin_model, validate_robin_model, test_robin_model
from pipeline.evaluate_robin import predict_answers_and_evaluate

from data_models.data_qa import train_data_preprocess


def get_args():

    parser = argparse.ArgumentParser(description='PyTorch RoBIn Training')

    # parameters
    parser.add_argument("--model_name", default='distilbert-base-uncased', type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--data_folder", default='./data', type=str,
                        help="Path to data folder.")
    parser.add_argument("--dataset_name", default='robqa', type=str,
                        help="Name of the dataset to use.")
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
    parser.add_argument("--pre_trained", default=False, action='store_true',
                        help="Use pretrained model.")
    parser.add_argument("--pos_weight", default=1.0, type=float,
                        help="Weight for positive class in BCE loss.")
    parser.add_argument("--lambda_p", default=0.5, type=float,
                        help="Weight for QA loss in total loss.")
    parser.add_argument("--qa_attention", default=False, action='store_true',
                        help="Use QA attention.")
    parser.add_argument("--mlp_classifier", default=False, action='store_true',
                        help="Use MLP classifier.")
    parser.add_argument("--loss_fn_cls", default='bce', type=str,
                        help="Loss function for classification.")

    # distributed training parameters
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed value (default: None)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Number of workers for the data loader.")

    parser.add_argument("--ckpt", default=None, type=str,
                        help="Path to checkpoint to load.")
    parser.add_argument("--resume", action='store_true',
                        help="Resume training from checkpoint.")
    parser.add_argument("--evaluate", action='store_true',
                        help="Evaluate model on test set.")
    parser.add_argument("--last_epoch", action='store_true',
                        help="Evaluate last epoch model.")

    parser.add_argument("--print_freq", default=10, type=int,
                        help="Print frequency.")

    args = parser.parse_args()
    args.model_path = "./save/{}_models".format(args.model_name)

    args.model_filename = 'robin_{}_len_{}_bsz_{}_lr_{}_wd_{}_lmb_{}'.format(args.dataset_name,
                                                                             args.max_seq_length,
                                                                             args.batch_size,
                                                                             args.learning_rate,
                                                                             args.weight_decay,
                                                                             args.lambda_p)
    if args.qa_attention:
        args.model_filename += '_qa_attention'

    if args.mlp_classifier:
        args.model_filename += '_mlp_classifier'

    args.model_filename += '_' + args.loss_fn_cls


    args.start_epoch = 0
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.save_folder = os.path.join(args.model_path, args.model_filename)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
        os.makedirs(os.path.join(args.save_folder, 'last_epoch'), exist_ok=True)

    return args


def trainer_fn(args, model, tokenizer):
    """Trains and validates the model.

    Args:
        args: Argument parser containing training parameters.
        model: The model to be trained.
        tokenizer: Tokenizer for the model.
    """
    best_loss = None
    val_loss = None

    dataset = load_dataset('json', data_files={
        'train': os.path.join(args.data_folder, 'rob_dataset_train.json'),
        'validation': os.path.join(args.data_folder, 'rob_dataset_test.json')})

    train_dataset = RoBInDataset(dataset, tokenizer, max_seq_length=args.max_seq_length, mode="train")
    val_dataset = RoBInDataset(dataset, tokenizer, max_seq_length=args.max_seq_length, mode="validate")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=default_data_collator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=default_data_collator
    )

    total_steps = len(train_loader) * args.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)

    for epoch in range(args.start_epoch, args.epochs):
        try:
            time1 = time.time()
            train_loss, = train_robin_model(train_loader, model, optimizer, scheduler, epoch, args)
            time2 = time.time()
            print('Training epoch {}, total time {:.2f}, loss {:.7f}'.format(epoch, (time2 - time1), train_loss))

            val_time1 = time.time()
            val_loss, = validate_robin_model(val_loader, model, epoch, args)
            val_time2 = time.time()

            print('Validation epoch {}, total time {:.2f}, Loss {:.3f}'.format(epoch, (val_time2 - val_time1), val_loss))

            if best_loss is None or val_loss < best_loss:
                # save best model
                best_loss = val_loss
                save_model_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'val_loss': val_loss,
                    'optimizer': optimizer.state_dict(),
                }, args.save_folder)
                model.save_pretrained(os.path.join(args.save_folder, 'pretrained'))
                tokenizer.save_pretrained(os.path.join(args.save_folder, 'pretrained'))

            print('Best loss {:.7f}'.format(best_loss))

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

        except Exception as e:
            print('Error in training epoch {}'.format(epoch))
            save_model_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'val_loss': val_loss,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.save_folder, 'last_epoch'))

            model.save_pretrained(os.path.join(args.save_folder, 'last_epoch', 'pretrained'))
            tokenizer.save_pretrained(os.path.join(args.save_folder, 'last_epoch', 'pretrained'))

            raise e

    print('Done!')


def test_fn(args, model, tokenizer):
    dataset = load_dataset('json', data_files={
        'train': os.path.join(args.data_folder, 'rob_dataset_train.json'),
        'validation': os.path.join(args.data_folder, 'rob_dataset_test.json')})

    val_dataset = RoBInDataset(dataset, tokenizer, max_seq_length=args.max_seq_length, mode="test")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=default_data_collator)

    val_time1 = time.time()
    start_logits, end_logits, logits = test_robin_model(val_loader, model, 0, args)
    # calculating metrics
    answers, metrics_, cls_f1, cls_pr, cls_rc, cls_ac, cls_preds, cls_probs = predict_answers_and_evaluate(start_logits, end_logits, logits,
                                                                        val_dataset.data, dataset["validation"])

    output_path = str(os.path.join(args.save_folder))
    if args.last_epoch:
        output_path = str(os.path.join(args.save_folder, 'last_epoch'))
    with open(os.path.join(output_path, 'answers.json'), 'w') as f:
        f.write(json.dumps(answers, indent=4))
    with open(os.path.join(output_path, 'cls_preds.json'), 'w') as f:
        f.write(json.dumps(cls_preds, indent=4))
    with open(os.path.join(output_path, 'cls_probs.json'), 'w') as f:
        f.write(json.dumps(cls_probs, indent=4))

    val_time2 = time.time()
    print('Validation total time {:.2f}, F1 {:.3f}, Exact Match {:.3f}, RoB F1 {:.3f} Precision {:.3f} Recall {:.3f} Accuracy {:.3f}'.format((val_time2 - val_time1),
                                                                                              metrics_['f1'],
                                                                                              metrics_['exact_match'],
                                                                                              cls_f1,
                                                                                              cls_pr,
                                                                                              cls_rc,
                                                                                              cls_ac))
    print('Done!')


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
    if __args.pre_trained:
        if __args.last_epoch:
            __tokenizer__ = AutoTokenizer.from_pretrained(os.path.join(__args.save_folder, 'last_epoch', 'pretrained'))
            __model__ = RoBInClassifier.from_pretrained(os.path.join(__args.save_folder, 'last_epoch', 'pretrained'))
        else:
            __tokenizer__ = AutoTokenizer.from_pretrained(os.path.join(__args.save_folder, 'pretrained'))
            __model__ = RoBInClassifier.from_pretrained(os.path.join(__args.save_folder, 'pretrained'))
    else:
        __tokenizer__ = AutoTokenizer.from_pretrained(__args.model_name, clean_up_tokenization_spaces=True)
        __config__ = RoBInConfig(__args.model_name, pos_weight=__args.pos_weight, lambda_p=__args.lambda_p,
                                    dropout=__args.dropout, sep_token_id=__tokenizer__.sep_token_id,
                                    qa_attention=__args.qa_attention, loss_fn_cls=__args.loss_fn_cls,
                                    mlp_classifier=__args.mlp_classifier)
        __model__ = RoBInClassifier(__config__)

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
