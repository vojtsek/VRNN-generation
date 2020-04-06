import argparse
import json
import time
import os
import shutil

import yaml
from torchvision.transforms import Compose as TorchCompose
from torch.utils.data import DataLoader as TorchDataLoader
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.logging import TensorBoardLogger

from .dataset import DataReader, CamRestReader, MultiWOZReader,\
    Dataset, ToTensor, Padding, WordToInt, Embeddings, Delexicalizer

from .model import VRNN, EpochEndCb, checkpoint_callback


def main(flags):
    with open(flags.config, 'rt') as in_fd:
        config = yaml.load(in_fd, Loader=yaml.FullLoader)

    delexicalizer = Delexicalizer(config['data_dir'])
    if config['data_type'] == 'raw':
        with open(os.path.join(config['data_dir'], 'data.json'), 'rt') as in_fd:
            data = json.load(in_fd)
        if config['domain'] == 'camrest':
            reader = CamRestReader()
        elif config['domain'] == 'woz-hotel':
            reader = MultiWOZReader(['hotel'])
        data_reader = DataReader(data=data, reader=reader, delexicalizer=delexicalizer)
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
                            extern_vocab=[w for w, _ in data_reader.all_words.most_common(5000)])
    # embeddings.add_tokens_rnd(delexicalizer.all_tags)
    composed_transforms = TorchCompose([WordToInt(embeddings),
                                        Padding(embeddings.w2id[Embeddings.PAD],
                                                data_reader.max_dial_len,
                                                data_reader.max_turn_len + 1,
                                                data_reader.max_slu_len + 1),  # +1 for <EOS>
                                        ToTensor()])
    train_dataset = Dataset(data_reader.train_set, transform=composed_transforms)
    valid_dataset = Dataset(data_reader.valid_set, transform=composed_transforms)
    test_dataset = Dataset(data_reader.test_set, transform=composed_transforms)
    train_loader = TorchDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = TorchDataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = TorchDataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

    if flags.model_path is not None:
        checkpoint = torch.load(flags.model_path)
        model = VRNN(config, embeddings, train_loader, valid_loader, test_loader)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = VRNN(config, embeddings, train_loader, valid_loader, test_loader)
        callbacks = [EpochEndCb()]
        logger = TensorBoardLogger(os.path.join(output_dir, 'tensorboard'), name='model')
        trainer = pl.Trainer(
            min_epochs=85,
            max_epochs=100,
            callbacks=callbacks,
            logger=logger,
            show_progress_bar=True,
            checkpoint_callback=checkpoint_callback(os.path.join(output_dir, 'model')),
            early_stop_callback=EarlyStopping(patience=3),
            progress_bar_refresh_rate=1
        )
        trainer.fit(model)

    model.eval()
    loader = TorchDataLoader(valid_dataset, batch_size=1, shuffle=True)
    with open(os.path.join(output_dir, 'output_all.txt'), 'wt') as all_fd, \
            open(os.path.join(output_dir, 'system_out.txt'), 'wt') as system_fd, \
            open(os.path.join(output_dir, 'system_ground_truth.txt'), 'wt') as system_gt_fd, \
            open(os.path.join(output_dir, 'user_out.txt'), 'wt') as user_fd, \
            open(os.path.join(output_dir, 'user_ground_truth.txt'), 'wt') as user_gt_fd, \
            open(os.path.join(output_dir, 'nlu_out.txt'), 'wt') as nlu_fd, \
            open(os.path.join(output_dir, 'nlu_ground_truth.txt'), 'wt') as nlu_gt_fd, \
            open(os.path.join(output_dir, 'z_posterior.txt'), 'wt') as z_post_fd, \
            open(os.path.join(output_dir, 'z_prior.txt'), 'wt') as z_prior_fd:

        for d, val_batch in enumerate(loader):
            predictions = model.predict(val_batch, embeddings.id2w)
            assert len(predictions.all_user_predictions) ==\
                len(predictions.all_system_predictions) ==\
                len(predictions.all_z_samples)
            print(f'Dialogue {d+1}', file=all_fd)
            for i in range(len(predictions.all_user_predictions)):
                print(f'\tTurn {i+1}', file=all_fd)
                print(f'\t{" ".join(predictions.all_user_predictions[i])}', file=all_fd)
                print(f'\t{" ".join(predictions.all_nlu_predictions[i])}', file=all_fd)
                print(f'\t{" ".join(predictions.all_system_predictions[i])}', file=all_fd)
                print(f'\tORIG:', file=all_fd)
                print(f'\t{" ".join(predictions.all_user_gt[i])}', file=all_fd)
                print(f'\t{" ".join(predictions.all_nlu_gt[i])}', file=all_fd)
                print(f'\t{" ".join(predictions.all_system_gt[i])}', file=all_fd)
                print(f'\tprior Z: {" ".join([str(z) for z in predictions.all_p_z_samples_matrix[i][0]])}', file=all_fd)
                print(f'\tpost Z: {" ".join([str(z) for z in predictions.all_q_z_samples_matrix[i][0]])}', file=all_fd)
                print('-' * 80, file=all_fd)

                print(" ".join(predictions.all_user_predictions[i]), file=user_fd)
                print(" ".join(predictions.all_system_predictions[i]), file=system_fd)
                print(" ".join(predictions.all_user_gt[i]), file=user_gt_fd)
                print(" ".join(predictions.all_system_gt[i]), file=system_gt_fd)
                print(" ".join(predictions.all_nlu_predictions[i]), file=nlu_fd)
                print(" ".join(predictions.all_nlu_gt[i]), file=nlu_gt_fd)
                print(" ".join([str(z) for z in predictions.all_p_z_samples_matrix[i][0]]), file=z_post_fd)

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
    args = parser.parse_args()

    main(args)
