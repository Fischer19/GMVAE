import torch
class newsgroupDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, train = True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        if self.trian:
            with open(root_dir, "rb") as f:
                dic = pickle.load(f)
                self.X = dic["X"][:12000]
                self.y = dic["y"][:12000]
        else:
            with open(root_dir, "rb") as f:
                dic = pickle.load(f)
                self.X = dic["X"][12000:]
                self.y = dic["y"][12000:]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx], self.y[idx]