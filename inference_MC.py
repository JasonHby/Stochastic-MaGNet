import torch
from torch.utils.data import DataLoader
from MaGNet import MaGNet
from Dataset import StockDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
dropout = 0.1

num_mc_runs = 100

print("Loading data...")
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

N = data.shape[0]
F = data.shape[2]

print(f"N = {N}, F = {F}")

# -----------------------------
# rebuild model
# -----------------------------
print("Building model...")
model = MaGNet(
    N, T, F, dim, num_MAGE, num_experts,
    num_heads_mha, num_F2DAttn, num_channels,
    num_heads_CausalMHA, num_TCH, TopK, M1,
    num_S2DAttn, num_GPH, M2,
    device=device,
    dropout=dropout
).to(device)

model.load_state_dict(torch.load(weight_path, map_location=device))

# use model.train instead of the evaluation so still have the MC dropout
model.train()

all_labels = []
for _, y in test_loader:
    all_labels.append(y)

all_labels = torch.cat(all_labels, dim=0)

# -----------------------------
# MC Dropout inference
# -----------------------------
print(f"Running MC Dropout inference for {num_mc_runs} runs...")

all_predictions = []

with torch.no_grad():
    for run in range(num_mc_runs):
        run_predictions = []

        for x, _ in test_loader:
            output, *_ = model(x)                 
            prob = torch.softmax(output, dim=-1)  
            run_predictions.append(prob)

        run_predictions = torch.cat(run_predictions, dim=0)
        all_predictions.append(run_predictions)

        if (run + 1) % 10 == 0:
            print(f"Completed {run + 1}/{num_mc_runs} runs")

all_predictions = torch.stack(all_predictions, dim=0)


# -----------------------------
# mean and variance
# -----------------------------
mean_pred = all_predictions.mean(dim=0)
var_pred = all_predictions.var(dim=0)

print("mean_pred shape:", mean_pred.shape)
print("var_pred shape:", var_pred.shape)

# -----------------------------
# final predicted class
# -----------------------------
final_pred = mean_pred.argmax(dim=-1)

# uncertainty score:
# average variance across classes
uncertainty_score = var_pred.mean(dim=-1)

print("final_pred shape:", final_pred.shape)
print("uncertainty_score shape:", uncertainty_score.shape)

# The accuracy from the mean prediction
accuracy = (final_pred == all_labels).float().mean().item() 
print(f"MC Dropout Test Accuracy: {accuracy:.4f}")

print("\nExample outputs:")
print("Mean prediction (first 5):")
print(mean_pred[:5])

print("\nVariance (first 5):")
print(var_pred[:5])

print("\nPredicted class (first 20):")
print(final_pred[:20])

print("\nTrue labels (first 20):")
print(all_labels[:20])

print("\nUncertainty score (first 20):")
print(uncertainty_score[:20])
