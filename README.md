# Stochastic MaGNet
Optimization study of the MaGNet algorithm from https://github.com/PeilinTime/MaGNet with Stochastic algorithms implementation

### Model Variants
| Version | MC Dropout placement |
|---------|----------------------|
| Magnetv1 | After MAGE + before output |
| Magnetv2 | After F2DAttn + before output |
| Magnetv3 | After MAGE, after F2DAttn, and before output |

### Training 
**Prerequisites**
- Python environment with PyTorch, torcheval, transformers installed
- `my_nas100_2025_data.pt` in the project root (generate via `data_collection/Nasdaq100_data_collection_and_cleaning.ipynb`)

**Steps**
1. Open `train.py` and change model version:
   ```python
   MODEL_VERSION = 'Magnetv2'  # or 'Magnetv3'
   ```

2. Run training:
   ```bash
   python train.py
   ```

3. Outputs are saved to the project root with a version suffix:
   - `best_model_Magnetv2.pth` — best checkpoint (by val accuracy)
   - `final_model_Magnetv2.pth` — model at last epoch
   - `training_history_Magnetv2.pt` — loss/accuracy history
   - `training_history_Magnetv2.png` — training curves plot

4. For MC Dropout inference, set the same `MODEL_VERSION` in `inference_MC.py` and run:
   ```bash
   python inference_MC.py
   ```
   Output: `mc_results_Magnetv2.pt` with `mean_pred` and `var_pred`.

**Device**: automatically uses CUDA (if none, then → MPS (Apple Silicon) → CPU)


