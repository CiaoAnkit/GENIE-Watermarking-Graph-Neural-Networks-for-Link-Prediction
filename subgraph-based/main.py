import torch
import os.path
import sys
import random
import numpy as np
sys.path.append('%s/pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from pytorch_DGCNN.main import *
from util_functions import *
import pickle

# Set random seeds for reproducibility
seed_value = 42

# PyTorch
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

# Numpy
np.random.seed(seed_value)

# Python
random.seed(seed_value)

torch.backends.cudnn.deterministic = True

data_name = sys.argv[1]

def split_data(data, split_ratio=0.5):
    random.shuffle(data)
    split_point = int(len(data) * split_ratio)
    return data[:split_point], data[split_point:]

# Loading hyperparameters
with open('./trained_models/{}/hyper.pkl'.format(data_name), 'rb') as file:
    hyperparameters = pickle.load(file)

model = Classifier(hyperparameters)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

state_dict = torch.load('./trained_models/{}/model.pth'.format(data_name), map_location=device)
model.load_state_dict(state_dict)

# Model evalutation before pruning
with open('./trained_models/{}/watermark_graphs.pkl'.format(data_name), 'rb') as file:
    # Load the object from the file
    val_watermark_graphs = pickle.load(file)

with open('./trained_models/{}/test_graphs.pkl'.format(data_name), 'rb') as file:
    # Load the object from the file
    test_graphs = pickle.load(file)

model.eval()

print("Model:SEAL Dataset:{}".format(data_name))

test_loss = loop_dataset(test_graphs, model, list(range(len(test_graphs))))
val_watermark_loss = loop_dataset(val_watermark_graphs, model, list(range(len(val_watermark_graphs))))

print("Test AUC:{} Watermark AUC:{}".format(test_loss[2],val_watermark_loss[2]))