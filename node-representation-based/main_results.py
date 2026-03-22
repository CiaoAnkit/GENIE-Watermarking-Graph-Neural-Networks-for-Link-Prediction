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
from utils import test_wm

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

model_name = sys.argv[2]

with open('./trained_models/{}/{data_name}/hyperparameters_gcn.pkl'.format(model_name,data_name=data_name),'rb') as f:
    hyperparameters_gcn = pickle.load(f)
with open('./trained_models/{}/{data_name}/hyperparameters_mlp.pkl'.format(model_name,data_name=data_name),'rb') as f:
    hyperparameters_mlp = pickle.load(f)
with open('./trained_models/{}/{data_name}/split_edge.pkl'.format(model_name,data_name=data_name),'rb') as f:
    split_edge = pickle.load(f)
with open('./trained_models/{}/{data_name}/data.pkl'.format(model_name,data_name=data_name),'rb') as f:
    data = pickle.load(f)

if model_name == 'gcn':
    model = GCN(hyperparameters_gcn[0],hyperparameters_gcn[1],
                hyperparameters_gcn[2],hyperparameters_gcn[3],hyperparameters_gcn[4])
elif model_name == 'sage':
    model = SAGE(hyperparameters_gcn[0],hyperparameters_gcn[1],
                hyperparameters_gcn[2],hyperparameters_gcn[3],hyperparameters_gcn[4])
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
state_dict = torch.load('trained_models/{}/{data_name}/gcn_state_dict.pth'.format(model_name,data_name=data_name), map_location=device)
model.load_state_dict(state_dict)

predictor = LinkPredictor(hyperparameters_mlp[0],hyperparameters_mlp[1],
            hyperparameters_mlp[2],hyperparameters_mlp[3],hyperparameters_mlp[4])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictor = predictor.to(device)
state_dict = torch.load('trained_models/{}/{data_name}/predictor_state_dict.pth'.format(model_name,data_name=data_name), map_location=device)
predictor.load_state_dict(state_dict)

model.eval()
predictor.eval()

test_hits_before = []
watermark_hits_before = []
test_auc_before,watermark_auc_before = test_wm(model,predictor,data,split_edge,evaluator,64 * 1024)

if model_name == 'gcn':
    print('Model:GCN Dataset:{}'.format(data_name))
elif model_name == 'sage':
    print('Model:GraphSAGE Dataset:{}'.format(data_name))
print(f'Test AUC: {100 * test_auc_before:.2f}%, '
    f'Watermark AUC: {100 * watermark_auc_before:.2f}%')

