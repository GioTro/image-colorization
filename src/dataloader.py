import torch
from pathlib import Path
from utils import Utils

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.util = Utils()

    def __getitem__(self, i):
        im = self.util.load_im(str(self.dataset[i]))
        X, y = self.util.preprocess_im(im)
        breakpoint
        return X, y

    def __len__(self):
        return len(self.dataset)

def prepare(set_spec, params):
    """params = (batch_size, num_workers, shuffle)"""
    X = list(set_spec.glob("**/*.JPEG"))
    batch_size, num_workers, shuffle = params

    loader = torch.utils.data.DataLoader(
        Dataset(X),
        drop_last=False,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )

    return loader


def return_loaders(batch_size=64, num_workers=5, shuffle=True):
    root = Path(__file__).parent.absolute()

    paths = [
        root / "../dataset/test/",
        root / "../dataset/train/",
        root / "../dataset/val/",
    ]

    paths = map(Path, paths)

    dataset = {}
    params=(batch_size, num_workers, shuffle)

    dataset["test"], dataset["train"], dataset["validation"] = map(
        lambda x : prepare(x, params=params), paths
    )

    return dataset