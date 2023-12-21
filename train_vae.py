import os
import argparse
import random
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from vae import VAE


class Trainer:
    def __init__(self, model, opt_kwargs=None, early_stopping=100, log_every=50, device="cpu", save_dir="./"):
        if opt_kwargs is None:
            opt_kwargs = {}
        self.model = model.to(device)
        opt_kwargs["lr"] = opt_kwargs.get("lr", 3e-4)
        self.opt = torch.optim.Adam(model.parameters(), **opt_kwargs)
        self.early_stopping = early_stopping
        self.log_every = log_every
        self.device = device
        self.save_ckpt_path = f"{save_dir}/checkpoint.pt"
        self.save_logs_path = f"{save_dir}/logs.txt"
        open(self.save_logs_path, "w").close()

    
    def train(self, train_loader):
        loss_prev, es_counter = None, 0
        for i, x in enumerate(train_loader):
            loss = self._train_step(x.to(self.device))
            if i % self.log_every == 0:
                with open(self.save_logs_path, "a") as file:
                    file.write(f"{loss}\n")
            if loss_prev is None:
                loss_prev = loss
                continue
            if loss_prev <= loss:
                es_counter += 1
            else:
                es_counter += 1
                torch.save(self.model.state_dict(), self.save_ckpt_path)
            if es_counter == self.early_stopping:
                break

    def _train_step(self, x):
        self.opt.zero_grad()
        loss = self.model.loss(x)
        loss.backward()
        self.opt.step()
        return loss.item()


class TestDataset(Dataset):
    def __init__(self, size):
        self.data = torch.rand(size, 2)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class MNISTDataset(Dataset):
    def __init__(self, train=True, with_targets=False, size=None):
        ds = torchvision.datasets.MNIST(root="./", train=train, download=True)
        self.data = ds.data.numpy().astype("float32")
        mu, std = np.mean(self.data), np.std(self.data)
        self.data = (self.data - mu) / std
        if size is None:
            size = len(self.data)
        self.data = self.data[:size].reshape(size, -1)

        self.with_targets = with_targets
        if with_targets:
            self.target = ds.targets.numpy()[:size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])
        if self.with_targets:
            y = torch.tensor(self.target[idx])
            return x, y
        return x
    
class Dataloader(DataLoader):
    def __init__(self, ds_name, *args, ds_kwargs=None, **kwargs):
        if ds_kwargs is None:
            ds_kwargs = {}
        if ds_name == "test":
            dataset = TestDataset(**ds_kwargs)
        elif ds_name == "mnist":
            dataset = MNISTDataset(**ds_kwargs)
        else:
            raise ValueError("There is not such a dataset")
        super().__init__(dataset, *args, **kwargs)
        self.ds_loader = super().__iter__()

    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            batch = next(self.ds_loader)
        except StopIteration:
            self.ds_loader = super().__iter__()
            batch = next(self.ds_loader)
        return batch


def train_vae(
    ds_name,
    in_dim,
    latent_dim,
    hidden_dim = None,
    train_size = None,
    device = "cpu",
    save_dir = None,
):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if save_dir is None:
        save_dir = f"./{ds_name}_{in_dim}_{latent_dim}_{hidden_dim}"
        if train_size is not None:
            save_dir += f"_{train_size}"

    loader_kwargs = {}
    loader_kwargs["batch_size"] = 256
    loader_kwargs["shuffle"] = True

    train_kwargs = {}
    train_kwargs["log_every"] = 5
    train_kwargs["early_stopping"] = 1000
    train_kwargs["device"] = device
    train_kwargs["save_dir"] = save_dir
    if not os.path.exists(train_kwargs["save_dir"]):
        os.mkdir(train_kwargs["save_dir"])

    loader = Dataloader(
        ds_name,
        ds_kwargs={"size": train_size},
        **loader_kwargs,
    )
    model = VAE(in_dim, latent_dim, hidden_dim)
    trainer = Trainer(model, **train_kwargs)
    trainer.train(loader)
    print(f"output_dir: {save_dir}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds_name", type=str)
    parser.add_argument("in_dim", type=int)
    parser.add_argument("latent_dim", type=int)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()
    train_vae(**vars(args))