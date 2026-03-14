import torch
from torch.utils.data import DataLoader
from MaGNet import MaGNet
from Dataset import StockDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# same settings as train.py
data_path = "djia_alpha158_alpha360.pt"
weight_path = "best_model.pth"

dim = 32
num_experts = 4
num_heads_mha = 2
num_channels = 4
num_heads_CausalMHA = 2
T = 20
batch_size = 32
num_MAGE = 2
num_F2DAttn = 1
num_TCH = 1
TopK = 32
M1 = 32
num_S2DAttn = 1
num_GPH = 1
M2 = 16
num_mc_runs = 100

# load data
data = torch.load(data_path).to(device)

total_date = data.shape[1]
train_cutoff = int(total_date * 0.7)
valid_cutoff = train_cutoff + int(total_date * 0.1)

test_data = data[:, valid_cutoff:]

epsilon = 1e-6
test_data_mean = test_data.mean(dim=1, keepdim=True)
test_data_std = test_data.std(dim=1, keepdim=True)
test_data = (test_data - test_data_mean) / (test_data_std + epsilon)

test_dataset = StockDataset(test_data, T, device)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# rebuild model
N = data.shape[0]
F = data.shape[2]

model = MaGNet(
    N, T, F, dim, num_MAGE, num_experts,
    num_heads_mha, num_F2DAttn, num_channels,
    num_heads_CausalMHA, num_TCH, TopK, M1,
    num_S2DAttn, num_GPH, M2,
    device=device,
    dropout=0.1
).to(device)

model.load_state_dict(torch.load(weight_path, map_location=device))

# keep dropout ON
model.train()

all_mc_preds = []

with torch.no_grad():
    for _ in range(num_mc_runs):
        one_run_preds = []

        for x, _ in test_loader:
            output, *_ = model(x)
            prob = torch.softmax(output, dim=-1)
            one_run_preds.append(prob)

        one_run_preds = torch.cat(one_run_preds, dim=0)
        all_mc_preds.append(one_run_preds)

all_mc_preds = torch.stack(all_mc_preds, dim=0)

mean_pred = all_mc_preds.mean(dim=0)
var_pred = all_mc_preds.var(dim=0)

print("mean_pred shape:", mean_pred.shape)
print("var_pred shape:", var_pred.shape)

torch.save(
    {
        "mean_pred": mean_pred.cpu(),
        "var_pred": var_pred.cpu(),
    },
    "mc_results.pt"
)

print("Saved to mc_results.pt")
