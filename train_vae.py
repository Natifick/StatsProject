import os
import argparse
import random
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from vae import VAE


class Trainer:
    def __init__(self, model, opt_kwargs=None, early_stopping=1, device="cpu", save_dir="./"):
        if opt_kwargs is None:
            opt_kwargs = {}
        self.model = model.to(device)
        opt_kwargs["lr"] = opt_kwargs.get("lr", 3e-4)
        self.opt = torch.optim.Adam(model.parameters(), **opt_kwargs)
        self.early_stopping = early_stopping
        self.device = device
        self.save_ckpt_path = f"{save_dir}/checkpoint.pt"
        self.save_logs_path = f"{save_dir}/logs.txt"
        open(self.save_logs_path, "w").close()
    
    def train(self, train_loader, n_epochs=None, eval_loader=None):
        if n_epochs is None:
            n_epochs = 1024

        loss_prev, es_counter = None, 0
        for _ in range(n_epochs):
            loss = 0
            for i, x in enumerate(train_loader):
                loss += self._train_step(x.to(self.device))
            loss /= i

            if eval_loader is not None:
                loss = self.eval(eval_loader)

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

    def eval(self, loader):
        i, loss = 0, 0.
        for x in loader:
            x = x.to(self.device)
            loss += self._eval_step(x.to(self.device))
            i += 1
        return loss / i

    def _train_step(self, x):
        self.model.train()
        self.opt.zero_grad()
        loss = self.model.loss(x)
        loss.backward()
        self.opt.step()
        return loss.item()
    
    @torch.no_grad()
    def _eval_step(self, x):
        self.model.eval()
        loss = self.model.loss(x)
        return loss.item()


class TestDataset(Dataset):
    def __init__(self, size):
        self.data = torch.rand(size, 2)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class MNISTDataset(Dataset):
    MU = 0.13066062
    STD = 0.30810776
    VAL_SIZE = 5000

    def __init__(self, train=True, with_targets=False, size=None, is_val=False):
        ds = torchvision.datasets.MNIST(root="./", train=train, download=True)
        data = ds.data.numpy().astype("float32") / 255.
        data = (data - self.MU) / self.STD
        target = ds.targets.numpy()

        # idcs = np.random.randint(len(target), size=len(target))
        # data = data[idcs]
        # target = target[idcs]

        if is_val:
            assert train
            data = data[-self.VAL_SIZE:]
            target = target[-self.VAL_SIZE:]
        else:
            if size is None:
                size = len(data)
                if train:
                    size -= self.VAL_SIZE
            data = data[:size]
            target = target[:size]

        self.data = data.reshape(len(data), -1)
        if with_targets:
            self.target = target
        self.with_targets = with_targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx])
        if self.with_targets:
            y = torch.tensor(self.target[idx])
            return x, y
        return x


class FiniteDataloader(DataLoader):
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
        save_dir += "_0"
        if os.path.exists(save_dir):
            num = 0
            while os.path.exists(save_dir):
                tmp = save_dir.split("_")
                tmp[-1] = str(num)
                save_dir = "_".join(tmp)
                num += 1
            

    loader_kwargs = {}
    loader_kwargs["batch_size"] = 256
    loader_kwargs["shuffle"] = True

    train_kwargs = {}
    train_kwargs["early_stopping"] = 20
    train_kwargs["device"] = device
    train_kwargs["save_dir"] = save_dir
    if not os.path.exists(train_kwargs["save_dir"]):
        os.mkdir(train_kwargs["save_dir"])

    train_loader = FiniteDataloader(
        ds_name,
        ds_kwargs={"size": train_size},
        **loader_kwargs,
    )
    val_loader = FiniteDataloader(
        ds_name,
        ds_kwargs={"size": train_size, "is_val": True},
        **loader_kwargs,
    )
    model = VAE(in_dim, latent_dim, hidden_dim)
    trainer = Trainer(model, **train_kwargs)
    trainer.train(train_loader, eval_loader=val_loader)
    print(f"output_dir: {save_dir}")
    return save_dir

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