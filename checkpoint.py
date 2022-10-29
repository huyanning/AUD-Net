# coding=utf-8
import json
import os
import shutil
import errno
import os.path as osp
import torch
from torch.nn import Parameter
import sys


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_checkpoint(state, is_best, epoch, filepath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(filepath))
    # mkdir_if_missing(filepath)
    # filename = osp.join(filepath, 'checkpoint.pth.tar')
    filename = filepath + '{}.pth.tar'.format(epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, filepath + 'best.pth.tar')
        # shutil.copy(filename, osp.join(filepath, 'checkpoint_best.pth.tar'))


def load_checkpoint(model, optimizer, filepath, args, epoch):
    filename = filepath + '{}.pth.tar'.format(epoch)
    if osp.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        if args.gpu is None:
            checkpoint = torch.load(filename)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(filename, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            args.start_episode = checkpoint['episode']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.dump_path, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.dump_path))
        args.start_epoch = 0


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


