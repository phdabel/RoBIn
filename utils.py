import os
import argparse

import torch
import torch.nn as nn

OUTPUT_FILENAME = 'checkpoint.pth.tar'


def get_args():
    parser = argparse.ArgumentParser(description='Arguments for training and evaluating models')

    # Required parameters
    parser.add_argument("--model_name", default='pubmedsim', type=str,
                        help="Model identifier.")
    parser.add_argument("--task", default="semantic_similarity",
                        type=str, choices=["semantic_similarity", "regression"],
                        help="Task identifier.")
    parser.add_argument("--data_folder", default='../data_models', type=str,
                        help="Path to data folder.")
    parser.add_argument("--max_seq_length", default=128, type=int,
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

    parser.add_argument("--ckpt", default=None, type=str,
                        help="Path to checkpoint to load.")
    parser.add_argument("--resume", action='store_true',
                        help="Resume training from checkpoint.")
    parser.add_argument("--evaluate", action='store_true',
                        help="Evaluate model on test set.")


    # distributed training parameters
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed value (default: None)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of data loading workers (default: 1)')


    args = parser.parse_args()
    args.model_path = "../save/{}_models".format(args.model_name)
    args.model_name = '{}_tsk_{}_len_{}_bsz_{}_lr_{}'.format(args.model_name,
                                                             args.task,
                                                             args.max_seq_length,
                                                             args.batch_size,
                                                             args.learning_rate)

    args.start_epoch = 0
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.save_folder = os.path.join(args.model_path, args.model_name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    return args


def load_state(args, model):
    if args.ckpt is not None:
        model_ckpt = torch.load(args.ckpt, map_location=args.device)
        model.load_state_dict(model_ckpt['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.ckpt, model_ckpt['epoch']))
    else:
        try:
            model_ckpt = torch.load(str(os.path.join(args.save_folder, OUTPUT_FILENAME)),
                                    map_location=args.device)
            model.load_state_dict(model_ckpt['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(os.path.join(args.save_folder, OUTPUT_FILENAME),
                                                                model_ckpt['epoch']))
        except FileNotFoundError:
            raise "No checkpoint found. Please specify a checkpoint to load or ensure a checkpoint exists at {}".format(
                args.save_folder)


def save_model_checkpoint(state, save_folder, filename=OUTPUT_FILENAME):
    filename = os.path.join(save_folder, filename)
    torch.save(state, filename)


def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
