import argparse
import json
import time
import os
import sys
import pickle
import shutil
import random
import wandb
from copy import deepcopy

import yaml
from torchvision.transforms import Compose as TorchCompose
from torch.utils.data import DataLoader as TorchDataLoader
import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.logging import TensorBoardLogger

from .evaluation import ZInfoEvaluator, BleuEvaluator
from .dataset import DataReader, CamRestReader, MultiWOZReader, SMDReader, \
    Dataset, ToTensor, Padding, WordToInt, Embeddings, Delexicalizer, Turn
from .utils import compute_ppl
from .model import VRNN, EpochEndCb, checkpoint_callback

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


def _parse_from_arg(arg):
    val = arg
    parsed = False
    if '.' in val: # could be float
        try:
            val = float(val)
            parsed = True
        except:
            pass # remain string
    else: # not float
        try:
            val = int(val) # could be int
            parsed = True
        except:
            pass
    if not parsed: # not int nor float
        if val.lower() == 'false':
            val = False
        elif val.lower() == 'true':
            val = True
    return val


def main(flags, config, config_path):
    for fl in vars(flags):
        if fl in config:
            val = getattr(flags, fl)
            if val is not None:
                config[fl] = _parse_from_arg(val)
    if config['delex']:
        delexicalizer = Delexicalizer(config['data_dir'])
    else:
        delexicalizer = None
    if not os.path.isdir(flags.output_dir):
        os.mkdir(flags.output_dir)

    sets = ['test', 'train', 'valid']
    readers = {}
    if config['domain'] == 'camrest':
        reader = CamRestReader()
    elif config['domain'] == 'woz-hotel':
        reader = MultiWOZReader(['hotel'], max_allowed_len=config['max_allowed_turn_len'])
    elif config['domain'] == 'smd':
        reader = SMDReader()
    elif config['domain'] == 'daily':
        reader = DDReader()
    for data_set in sets:
        with open(os.path.join(config['data_dir'], f'{data_set}.json'), 'rt') as in_fd:
            data = json.load(in_fd)

        data_reader = DataReader(data=data,
                                 reader=reader,
                                 delexicalizer=delexicalizer,
                                 db_file=os.path.join(config['data_dir'], 'db.json'),
                                 train=1, valid=0)
        enchance_reader_with_api(data_reader, config['domain'])
        for d in data_reader.dialogues:
            for t in d.turns:
                print(t.user, t.system)

def enchance_reader_with_api(data_reader, domain):
    enhanced = []
    for dial in data_reader.dialogues:
        new_dial = deepcopy(dial)
        new_i = 0
        for i, turn in enumerate(dial.turns):
            if should_include_result(turn.system, turn.db_result, domain):
                db_turn = create_db_turn(turn, domain)
                if db_turn.system is None:
                    new_i += 1
                    continue
                new_turn = deepcopy(turn)
                new_turn.user = db_turn.system
                new_dial.turns[new_i].system = db_turn.user
                new_i += 1
                new_dial.turns.insert(new_i, new_turn)
            new_i += 1
        enhanced.append(new_dial)
        data_reader.max_dial_len = max(data_reader.max_dial_len, len(new_dial.turns) * 2)
        data_reader.max_turn_len = max(data_reader.max_turn_len, max([max(len(t.user), len(t.system)) for t in new_dial.turns]))
    data_reader._dialogues = enhanced
    return enhanced


def should_include_result(system, db_result, domain):
    system = ' '.join(system).lower()
    if domain == 'camrest':
        return any(['there are {n}' in system for n in ['no', '<num>', 'two', 'three', 'four', 'five']]) or\
            'there is not' in system or \
            any([res['name'] in system for res in db_result]) or \
            '<name>' in system
    if domain == 'smd':
        return True
    if domain == 'woz-hotel':
        return True
        def _get_values(res):
            for key, val in res.items():
                if isinstance(val, list):
                    val = ','.join([str(x) for x in val])
                val = str(val).lower()
                yield val

            return any([val.lower() in system for res in db_result for val in _get_values(res)]) or \
                '<name>' in system


def create_db_turn(turn, domain):
    def _format_result(system, db_res):
        if domain == 'camrest':
            if len(db_res) == 0:
                return ['no', 'results']
            used_result = db_res[0]
            for res in db_res:
                if res['name'] in ' '.join(system):
                    used_result = res

            name = used_result['name'] if 'name' in used_result else ''
            address = used_result['address'] if 'address' in used_result else ''
            phone = used_result['phone'] if 'phone' in used_result else ''
            postcode = used_result['postcode'] if 'postcode' in used_result else ''
            return f'result: {name} , {address} , {phone} , {postcode}'.split()
        else:
            if len(db_res) == 0:
                return None
            used_result = db_res[0]
            for res in db_res:
                for k, v in res.items():
                    if isinstance(v, list):
                        v = ','.join([str(i) for i in v])
                    if not isinstance(v, str):
                        continue
                    if v in ' '.join(system):
                        used_result = res


            return [v + " , " for v in used_result.values() if isinstance(v, str)]

    db_turn = Turn()
    db_turn.usr_slu = []
    db_turn.sys_slu = []
    if turn.state is not None:
        db_turn.user = [slot.val for slot in turn.state]
    else:
        db_turn.user = [slot.val for slot in turn.usr_slu]
    db_turn.system = _format_result(turn.orig_system, turn.db_result)
    return db_turn


if __name__ == '__main__':
    config_path = sys.argv[1]
    for i in range(2,len(sys.argv)):
        sys.argv[i-1] = sys.argv[i]
    del sys.argv[-1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)

    with open(config_path, 'rt') as in_fd:
        config = yaml.load(in_fd, Loader=yaml.FullLoader)
    for key in config.keys():
        parser.add_argument(f'--{key}', type=str, required=False, default=None)
    args = parser.parse_args()
    main(args, config, config_path)
