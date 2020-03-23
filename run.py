import argparse
import json

import yaml
from torchvision.transforms import Compose as TorchCompose
from torch.utils.data import DataLoader as TorchDataLoader
import pytorch_lightning as pl

from .dataset.datareader import DataReader, CamRestReader, MultiWOZReader
from .dataset.dataset import Dataset, ToTensor, Padding, WordToInt
from .dataset.embedding import Embeddings
from .model.vrnn import VRNN


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

    embeddings = Embeddings(config['embedding_fn'])
    composed_transforms = TorchCompose([WordToInt(embeddings),
                                        Padding(embeddings.w2id[Embeddings.PAD],
                                                data_reader.max_dial_len,
                                                data_reader.max_turn_len),
                                        ToTensor()])
    train_dataset = Dataset(data_reader.train_set, transform=composed_transforms)
    valid_dataset = Dataset(data_reader.valid_set, transform=composed_transforms)
    test_dataset = Dataset(data_reader.test_set, transform=composed_transforms)
    train_loader = TorchDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = TorchDataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = TorchDataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)
    model = VRNN(config, embeddings, train_loader, valid_loader, test_loader)
    trainer = pl.Trainer()
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', type=str)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()

    main(args)
