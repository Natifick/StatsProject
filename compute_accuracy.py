import argparse
import torch
from sklearn.linear_model import LogisticRegression
from vae import VAE
from train_vae import MNISTDataset
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", type=str)
args = parser.parse_args()

model_args = []
for i, arg in enumerate(args.model_dir.split("/")[-1].split("_")):
    if i == 0:
        ds_name = arg
        assert ds_name == "mnist"
        continue
    if arg != "None":
        model_args.append(int(arg))
    else:
        model_args.append(None)

if len(model_args) > 3:
    model_args = model_args[:3]

model = VAE(*model_args)
model.load_state_dict(
    torch.load(f"{args.model_dir}/checkpoint.pt")
)

train_ds = MNISTDataset(train=True, with_targets=True)
X_train = torch.tensor(train_ds.data)
Z_train = model.encode(X_train).cpu().numpy()
linreg = LogisticRegression(random_state=0).fit(Z_train, train_ds.target)

test_ds = MNISTDataset(train=False, with_targets=True)
X_test = torch.tensor(test_ds.data)
Z_test = model.encode(X_test).cpu().numpy()
pred = linreg.predict(Z_test)

output_file = f"{args.model_dir}/accuracy.txt"
with open(output_file, "w") as file:
    file.write(f"{accuracy_score(test_ds.target, pred)}\n")
print(f"output_file: {output_file}")
