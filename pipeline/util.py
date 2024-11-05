import os


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.short_name = name.lower().replace(' ', '_')
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __csv__(self):
        sep = ';'
        fmtstr = sep.join(['{short_name}',
                           '{val' + self.fmt + '}'])
        return fmtstr.format(**self.__dict__).split(sep)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logfile=""):
        self.batch_fmtstr = ProgressMeter._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logfile = logfile
        self.header = False

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def log_metrics(self, batch, sep=";"):
        epoch = int(self.prefix[8:-1])
        entries = dict()
        entries['epoch'] = epoch
        entries['batch'] = batch
        for i, meter in enumerate(self.meters):
            metric, value = meter.__csv__()
            entries[metric] = float(value)

        _logfilename_ = os.path.abspath(os.path.normpath(self.logfile))
        try:
            _logfile = open(_logfilename_, 'a')
        except FileNotFoundError:
            os.makedirs(os.path.dirname(_logfilename_), exist_ok=True)
            _logfile = open(_logfilename_, 'w+')

        if not self.header and epoch == 0:
            self.header = sep.join([key for key in entries.keys()]) + '\n'
            _logfile.write(self.header)
            self.header = True

        content = sep.join([str(entries[key]) for key in entries.keys()]) + '\n'
        _logfile.write(content)
        _logfile.close()

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def shift_tokens_right(input_ids, pad_token_id):
  """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
  """
  prev_output_tokens = input_ids.clone()
  index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
  prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
  prev_output_tokens[:, 1:] = input_ids[:, :-1]
  return prev_output_tokens


def create_metrics_object():
    return {
        "source": [],
        "target": [],
        "output": [],
        "bleu": [],
        "sari": [],
        "fkgl": [],
        "perplexity": []
    }.copy()