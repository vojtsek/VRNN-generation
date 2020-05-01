import argparse
import json
import time
import os
import sys
import shutil
import random

import yaml
from torchvision.transforms import Compose as TorchCompose
from torch.utils.data import DataLoader as TorchDataLoader
import pytorch_lightning as pl
import torch
import numpy
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.logging import TensorBoardLogger
from git import Repo

from .dataset import DataReader, CamRestReader, MultiWOZReader, SMDReader, \
    Dataset, ToTensor, Padding, WordToInt, Embeddings, Delexicalizer

from .model import VRNN, EpochEndCb, checkpoint_callback

seed = 42
numpy.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


def main(flags):
    with open(flags.config, 'rt') as in_fd:
        config = yaml.load(in_fd, Loader=yaml.FullLoader)

    delexicalizer = Delexicalizer(config['data_dir'])

    sets = ['test', 'train', 'valid']
    readers = {}
    if config['domain'] == 'camrest':
        reader = CamRestReader()
    elif config['domain'] == 'woz-hotel':
        reader = MultiWOZReader(['hotel'])
    elif config['domain'] == 'smd':
        reader = SMDReader()
    for data_set in sets:
        with open(os.path.join(config['data_dir'], f'{data_set}.json'), 'rt') as in_fd:
            data = json.load(in_fd)

        readers[data_set] = DataReader(data=data,
                                       reader=reader,
                                       delexicalizer=delexicalizer,
                                       db_file=os.path.join(config['data_dir'], 'db.json'),
                                       train=1, valid=0)

    if args.output_dir is None:
        print('Output directory not provided, exiting.')
        return
    output_dir = os.path.join(args.output_dir, f'run_{int(time.time())}')
    if args.train_more:
        output_dir += '_retrain'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    shutil.copy(flags.config, os.path.join(output_dir, 'conf.yaml'))
    repo = Repo(os.path.dirname(sys.argv[0]))
    with open(os.path.join(output_dir, 'gitcommit.txt'), 'wt') as fd:
        print(f'{repo.head.commit}@{repo.active_branch}', file=fd)

    embeddings = Embeddings(config['embedding_fn'],
                            out_fn='VRNN/data/embeddings/fasttext-wiki.pkl',
                            extern_vocab=[w for w, _ in (readers['train'].all_words.most_common(5000) +\
                                                         readers['valid'].all_words.most_common(5000))])
    embeddings.add_tokens_rnd(delexicalizer.all_tags)
    composed_transforms = TorchCompose([WordToInt(embeddings, config['db_cutoff']),
                                        Padding(embeddings.w2id[Embeddings.PAD],
                                                max(readers['train'].max_dial_len, readers['valid'].max_dial_len),
                                                max(readers['train'].max_turn_len, readers['valid'].max_turn_len) + 2,
                                                max(readers['train'].max_slu_len, readers['valid'].max_slu_len) + 2),
                                        # +2 for <BOS>,<EOS>
                                        ToTensor()])
    train_dataset = Dataset(readers['train'].dialogues, transform=composed_transforms)
    valid_dataset = Dataset(readers['valid'].dialogues, transform=composed_transforms)
    test_dataset = Dataset(readers['test'].dialogues, transform=composed_transforms)
    train_loader = TorchDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = TorchDataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = TorchDataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

    config['system_z_total_size'] = config['system_z_logits_dim'] * config['system_number_z_vectors']
    config['user_z_total_size'] = config['user_z_logits_dim'] * config['user_number_z_vectors']
    config['encoder_hidden_total_size'] = config['input_encoder_hidden_size'] * (1 + config['bidirectional_encoder'])
    if flags.model_path is not None:
        checkpoint = torch.load(flags.model_path)
        model = VRNN(config, embeddings, train_loader, valid_loader, test_loader)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = VRNN(config, embeddings, train_loader, valid_loader, test_loader)
    if flags.train_more or flags.model_path is None:
        config['retraining'] = flags.train_more
        callbacks = [EpochEndCb(), EvaluationCb(output_dir, valid_dataset)]
        logger = TensorBoardLogger(os.path.join(output_dir, 'tensorboard'), name='model')
        trainer = pl.Trainer(
            min_epochs=config['min_epochs'],
            max_epochs=config['max_epochs'],
            callbacks=callbacks,
            logger=logger,
            show_progress_bar=True,
            checkpoint_callback=checkpoint_callback(os.path.join(output_dir, 'model')),
            early_stop_callback=EarlyStopping(patience=3),
            progress_bar_refresh_rate=1
        )
        run_evaluation(output_dir, model, valid_dataset)
        trainer.fit(model)
        run_evaluation(output_dir, model, valid_dataset)


class EvaluationCb(pl.Callback):
    def __init__(self, output_dir, dataset):
        self.output_dir = output_dir
        self.dataset = dataset

    def on_epoch_end(self, trainer, model):
        run_evaluation(self.output_dir, model, self.dataset)
        model.train()


def run_evaluation(output_dir, model, dataset):
    model.eval()
    loader = TorchDataLoader(dataset, batch_size=1, shuffle=True)
    with open(os.path.join(output_dir, f'output_all_{model.epoch_number}.txt'), 'wt') as all_fd, \
            open(os.path.join(output_dir, 'system_out.txt'), 'wt') as system_fd, \
            open(os.path.join(output_dir, 'system_ground_truth.txt'), 'wt') as system_gt_fd, \
            open(os.path.join(output_dir, 'user_out.txt'), 'wt') as user_fd, \
            open(os.path.join(output_dir, 'user_ground_truth.txt'), 'wt') as user_gt_fd, \
            open(os.path.join(output_dir, 'nlu_out.txt'), 'wt') as nlu_fd, \
            open(os.path.join(output_dir, 'nlu_ground_truth.txt'), 'wt') as nlu_gt_fd, \
            open(os.path.join(output_dir, 'z_posterior.txt'), 'wt') as z_post_fd, \
            open(os.path.join(output_dir, 'z_prior.txt'), 'wt') as z_prior_fd, \
            open(os.path.join(output_dir, 'z_user.txt'), 'wt') as z_user_fd:

        for d, val_batch in enumerate(loader):
            predictions = model.predict(val_batch)
            assert len(predictions.all_user_predictions) ==\
                len(predictions.all_system_predictions) ==\
                len(predictions.all_z_samples)
            print(f'Dialogue {d+1}', file=all_fd)
            for i in range(len(predictions.all_user_predictions)):
                print(f'\tTurn {i+1}', file=all_fd)
                print(f'\tUSER HYP:{" ".join(predictions.all_user_predictions[i])}', file=all_fd)
                print(f'\t{" ".join(predictions.all_usr_nlu_predictions[i])}', file=all_fd)
                print(f'\tSYS HYP:{" ".join(predictions.all_system_predictions[i])}', file=all_fd)
                print(f'\t{" ".join(predictions.all_sys_nlu_predictions[i])}', file=all_fd)
                print(f'\tORIG:', file=all_fd)
                print(f'\tUSER GT{" ".join(predictions.all_user_gt[i])}', file=all_fd)
                print(f'\t{" ".join(predictions.all_usr_nlu_gt[i])}', file=all_fd)
                print(f'\tSYS GT:{" ".join(predictions.all_system_gt[i])}', file=all_fd)
                print(f'\t{" ".join(predictions.all_sys_nlu_gt[i])}', file=all_fd)
                print(f'\tprior Z: {" ".join([str(z) for z in predictions.all_p_z_samples_matrix[i][0]])}', file=all_fd)
                print(f'\tpost Z: {" ".join([str(z) for z in predictions.all_q_z_samples_matrix[i][0]])}', file=all_fd)
                print(f'\tuser Z: {" ".join([str(z) for z in predictions.all_user_z_samples_matrix[i][0]])}', file=all_fd)
                print(f'\tdb: {predictions.db_data[i][0].item()}', file=all_fd)
                print('-' * 80, file=all_fd)

                print(" ".join(predictions.all_user_predictions[i]), file=user_fd)
                print(" ".join(predictions.all_system_predictions[i]), file=system_fd)
                print(" ".join(predictions.all_user_gt[i]), file=user_gt_fd)
                print(" ".join(predictions.all_system_gt[i]), file=system_gt_fd)
                print(" ".join(predictions.all_usr_nlu_predictions[i]), file=nlu_fd)
                print(" ".join(predictions.all_usr_nlu_gt[i]), file=nlu_gt_fd)
                print(" ".join([str(z) for z in predictions.all_q_z_samples_matrix[i][0]]), file=z_post_fd)
                print(" ".join([str(z) for z in predictions.all_p_z_samples_matrix[i][0]]), file=z_prior_fd)
                print(" ".join([str(z) for z in predictions.all_user_z_samples_matrix[i][0]]), file=z_user_fd)

            print('', file=user_fd)
            print('', file=system_fd)
            print('', file=user_gt_fd)
            print('', file=system_gt_fd)
            print('', file=z_post_fd)
            print('', file=z_prior_fd)
            print('=' * 80, file=all_fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--train_more', action='store_true')
    args = parser.parse_args()

    main(args)
