import argparse
import json
import time
import os
import shutil

import yaml
from torchvision.transforms import Compose as TorchCompose
from torch.utils.data import DataLoader as TorchDataLoader
import pytorch_lightning as pl

from .dataset.datareader import DataReader, CamRestReader, MultiWOZReader
from .dataset.dataset import Dataset, ToTensor, Padding, WordToInt
from .dataset.embedding import Embeddings
from .model.vrnn import VRNN, EpochEndCb


def main(flags):
    with open(flags.config, 'rt') as in_fd:
        config = yaml.load(in_fd, Loader=yaml.FullLoader)
    if config['data_type'] == 'raw':
        with open(config['data_fn'], 'rt') as in_fd:
            data = json.load(in_fd)
        if config['domain'] == 'camrest':
            reader = CamRestReader()
        elif config['domain'] == 'woz-hotel':
            reader = MultiWOZReader(['hotel'])
        data_reader = DataReader(data=data, reader=reader)
    else:
        data_reader = DataReader(saved_dialogues=config['data_fn'])

    if args.output_dir is None:
        print('Output directory not provided, exiting.')
        return
    output_dir = os.path.join(args.output_dir, f'run_{int(time.time())}')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    shutil.copy(flags.config, os.path.join(output_dir, 'conf.yaml'))

    embeddings = Embeddings(config['embedding_fn'],
                            out_fn='VRNN/data/embeddings/fasttext-wiki.pkl',
                            extern_vocab=list(data_reader.all_words.keys()))
    composed_transforms = TorchCompose([WordToInt(embeddings),
                                        Padding(embeddings.w2id[Embeddings.PAD],
                                                data_reader.max_dial_len,
                                                data_reader.max_turn_len + 1),  # +1 for <EOS>
                                        ToTensor()])
    train_dataset = Dataset(data_reader.train_set, transform=composed_transforms)
    valid_dataset = Dataset(data_reader.valid_set, transform=composed_transforms)
    test_dataset = Dataset(data_reader.test_set, transform=composed_transforms)
    train_loader = TorchDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = TorchDataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = TorchDataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)
    model = VRNN(config, embeddings, train_loader, valid_loader, test_loader)
    callbacks = [EpochEndCb()]
    trainer = pl.Trainer(
        min_epochs=10,
        max_epochs=50,
        callbacks=callbacks,
        show_progress_bar=True,
        progress_bar_refresh_rate=1
    )
    print(model)
    trainer.fit(model)
    model.eval()
    loader = TorchDataLoader(valid_dataset, batch_size=1, shuffle=True)
    with open(os.path.join(output_dir, 'output_all.txt'), 'wt') as all_fd, \
            open(os.path.join(output_dir, 'system_out.txt'), 'wt') as system_fd, \
            open(os.path.join(output_dir, 'system_ground_truth.txt'), 'wt') as system_gt_fd, \
            open(os.path.join(output_dir, 'user_out.txt'), 'wt') as user_fd, \
            open(os.path.join(output_dir, 'user_ground_truth.txt'), 'wt') as user_gt_fd, \
            open(os.path.join(output_dir, 'z_posterior.txt'), 'wt') as z_post_fd, \
            open(os.path.join(output_dir, 'z_prior.txt'), 'wt') as z_prior_fd:

        for d, val_batch in enumerate(loader):
            all_user_predictions, all_user_gt, all_system_predictions, all_system_gt, all_z_samples =\
                model.predict(val_batch, embeddings.id2w)
            assert len(all_user_predictions) == len(all_system_predictions) == len(all_z_samples)
            print(f'Dialogue {d+1}', file=all_fd)
            for i in range(len(all_user_predictions)):
                print(f'\tTurn {i+1}', file=all_fd)
                print(f'\t{" ".join(all_user_predictions[i])}', file=all_fd)
                print(f'\t{" ".join(all_system_predictions[i])}', file=all_fd)
                print(f'\tORIG:', file=all_fd)
                print(f'\t{" ".join(all_user_gt[i])}', file=all_fd)
                print(f'\t{" ".join(all_system_gt[i])}', file=all_fd)
                print(f'\tZ: {all_z_samples[i]}', file=all_fd)
                print('-' * 80, file=all_fd)

                print(" ".join(all_user_predictions[i]), file=user_fd)
                print(" ".join(all_system_predictions[i]), file=system_fd)
                print(" ".join(all_user_gt[i]), file=user_gt_fd)
                print(" ".join(all_system_gt[i]), file=system_gt_fd)
                print(all_z_samples[i][0], file=z_post_fd)

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
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    main(args)
