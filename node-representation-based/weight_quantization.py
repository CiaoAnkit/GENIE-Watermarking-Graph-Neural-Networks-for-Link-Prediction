import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import sys
import random
import numpy as np
import pickle
from ogb.linkproppred import Evaluator

# importing models
from models import GCN,SAGE,LinkPredictor
from utils import test_wm,test

# Set random seeds for reproducibility
seed_value = 1337

# PyTorch
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

# Numpy
np.random.seed(seed_value)

# Python
random.seed(seed_value)

# use when you're using cuda
torch.backends.cudnn.deterministic = True

evaluator = Evaluator(name='ogbl-collab')

data_name = sys.argv[1]

with open('./trained_models/gcn/{data_name}/hyperparameters_gcn.pkl'.format(data_name=data_name),'rb') as f:
    hyperparameters_gcn = pickle.load(f)
with open('./trained_models/gcn/{data_name}/hyperparameters_mlp.pkl'.format(data_name=data_name),'rb') as f:
    hyperparameters_mlp = pickle.load(f)
with open('./trained_models/gcn/{data_name}/split_edge.pkl'.format(data_name=data_name),'rb') as f:
    split_edge = pickle.load(f)
with open('./trained_models/gcn/{data_name}/data.pkl'.format(data_name=data_name),'rb') as f:
    data = pickle.load(f)

model = GCN(hyperparameters_gcn[0],hyperparameters_gcn[1],
            hyperparameters_gcn[2],hyperparameters_gcn[3],hyperparameters_gcn[4])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
state_dict = torch.load('trained_models/gcn/{data_name}/gcn_state_dict.pth'.format(data_name=data_name), map_location=device)
model.load_state_dict(state_dict)

predictor = LinkPredictor(hyperparameters_mlp[0],hyperparameters_mlp[1],
            hyperparameters_mlp[2],hyperparameters_mlp[3],hyperparameters_mlp[4])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictor = predictor.to(device)
state_dict = torch.load('trained_models/gcn/{data_name}/predictor_state_dict.pth'.format(data_name=data_name), map_location=device)
predictor.load_state_dict(state_dict)

model.eval()
predictor.eval()

# Stats before quantisation
# print("Before Quantisation")
test_auc_before,watermark_auc_before = test_wm(model,predictor,data,split_edge,evaluator,64 * 1024)

# print(f'Test AUC: {100 * test_auc_before:.2f}%, '
#     f'Watermark AUC: {100 * watermark_auc_before:.2f}%')

def quantize_weights(model, bits=3):
    scale = 2 ** bits - 1
    for param in model.parameters():
        if param.requires_grad:  # Only quantize trainable parameters
            with torch.no_grad():  # Ensure no gradient computation for these operations
                param_data_before = param.data
                min_val = param.data.min()
                max_val = param.data.max()
                
                # Check if min and max are the same (i.e., all parameter values are the same)
                if min_val == max_val:
                    # If all values are the same, quantization doesn't make sense
                    # You can skip quantization, or set quantized values to zero or any baseline value
                    quantized = torch.zeros_like(param.data)
                else:
                    # Proceed with quantization
                    quantized = torch.round((param.data - min_val) * scale / (max_val - min_val))
                    # Rescale quantized weights to original range
                    param.data = (quantized / scale) * (max_val - min_val) + min_val

                if torch.isnan(param.data).any():
                    print("NaN detected after quantization")
                    print("scale:", scale)
                    print("Original parameter data:", param_data_before)
                    print("min_val:", min_val)
                    print("max_val:", max_val)
                    print("Quantized data:", quantized)

def check_for_nan_in_parameters(model):
    for name, param in model.named_parameters():
        if torch.isnan(param.data).any():
            print(f"NaN found in {name}")
            param.data[torch.isnan(param.data)] = 0  # Replace NaNs with 0 or some appropriate value

# Weight Quantisation begins
quantize_weights(model)
check_for_nan_in_parameters(model)
quantize_weights(predictor)
check_for_nan_in_parameters(predictor)

# Stats after quantisation
# print("After Quantisation")
test_auc_after,watermark_auc_after = test_wm(model,predictor,data,split_edge,evaluator,64 * 1024)
# print(f'Test AUC: {100 * test_auc_after:.2f}%, '
#     f'Watermark AUC: {100 * watermark_auc_after:.2f}%')

with open('quantization_results.txt', 'a') as file:
    # Write Test AUC results
    file.write(f"Dataset:{data_name}\n")
    file.write("Test AUC\n")
    file.write("Before | After\n")
    file.write(f"{test_auc_before} | {test_auc_after}\n\n")

    # Write Watermark AUC results
    file.write("Watermark AUC\n")
    file.write("Before | After\n")
    file.write(f"{watermark_auc_before} | {watermark_auc_after}\n\n")
