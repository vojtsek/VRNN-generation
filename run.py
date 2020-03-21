import argparse
import json

from torchvision.transforms import Compose as TorchCompose
from torch.utils.data import DataLoader as TorchDataLoader
from .dataset.datareader import DataReader, CamRestReader, MultiWOZReader
from .dataset.dataset import Dataset, ToTensor, Padding, WordToInt
from .dataset.embedding import Embeddings


def main(flags):
    if flags.data_type == 'raw':
        with open(flags.data_fn, 'rt') as infd:
            data = json.load(infd)
        if flags.domain == 'camrest':
            reader = CamRestReader()
        elif flags.domain == 'woz-hotel':
            reader = MultiWOZReader(['hotel'])
        data_reader = DataReader(data=data, reader=reader)
    else:
        data_reader = DataReader(saved_dialogues=flags.data_fn)

    embeddings = Embeddings(flags.embedding_fn)
    composed_transforms = TorchCompose([WordToInt(embeddings),
                                        Padding(embeddings.w2id[Embeddings.PAD],
                                                data_reader.max_dial_len,
                                                data_reader.max_turn_len),
                                        ToTensor()])
    dataset = Dataset(data_reader, transform=composed_transforms)
    data_loader = TorchDataLoader(dataset, batch_size=4, shuffle=True)

    for i, sample_batch in enumerate(data_loader):
        print(i, sample_batch['dialogue'].shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_fn', type=str)
    parser.add_argument('--domain', type=str, default='camrest')
    parser.add_argument('--data_type', type=str, default='raw')
    parser.add_argument('--embedding_fn', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    main(args)
