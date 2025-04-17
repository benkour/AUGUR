# %%
import random
import os
import re
import numpy as np
import warnings
from sklearn.utils import shuffle
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoLoader
import gpytorch
from helpers import get_graph
from models import BaseGNN, ExactGPModel
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

warnings.filterwarnings("ignore", category=RuntimeWarning)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
# %%


def data_preparation(folder='/data_processed/', device=device, types_to_include=['Pt3', 'Pt6', 'Pt9']):
    data_list = []
    data_dict = {t: {} for t in types_to_include}
    root_path = os.getcwd() + folder
    filenames = os.listdir(root_path)
    filenames.sort()
    pattern = re.compile(r'(.*)_(\d+)_data_(-?\d+)\.pt')

    for filename in filenames:
        if filename.endswith('.pt'):
            match = pattern.match(filename)
            if match:
                type_key, file_index, data_index = match.groups()
                file_index = int(file_index)
                data_index = int(data_index)

                if type_key in types_to_include:
                    full_path = os.path.join(root_path, filename)
                    graph = torch.load(full_path, weights_only=False)
                    graph.x = graph.x.to(device)
                    graph.edge_attr = graph.edge_attr.to(device)
                    graph.edge_index = graph.edge_index.to(device)
                    graph.y = torch.tensor(
                        graph.y, dtype=torch.float).to(device)
                    data_list.append(graph)

                    if file_index not in data_dict[type_key]:
                        data_dict[type_key][file_index] = {}
                    if data_index not in data_dict[type_key][file_index]:
                        data_dict[type_key][file_index][data_index] = []
                    data_dict[type_key][file_index][data_index].append(graph)

    return data_list, data_dict


def atom_extraction(cluster_name='Pt3'):

    with open('data_processed/' + cluster_name + '_atoms.txt', 'r') as f:
        lines = f.readlines()
        atoms = [line.strip() for line in lines]
    return atoms


def train(train_loader, optimizer, device, model_gp, mll):

    model_gp.train()
    # "Loss" for GPs - the marginal log likelihood
    for data in train_loader:
        data = data.to(device)
        # zero gradients from previous iteration
        optimizer.zero_grad()
        # output from model
        output = model_gp(data)
        # calc loss and backprop gradients
        loss = -mll(output, data.y.reshape(-1,))
        loss.backward()
        optimizer.step()
    return loss / len(train_loader.dataset)


def validate(validation_loader, device, model_gp, likelihood, mll):
    model_gp.eval()
    likelihood.eval()

    with torch.no_grad():
        validation_loss = 0.0
        for data in validation_loader:
            data = data.to(device)
            output = likelihood(model_gp(data))
            # output = model_gp(data)

            loss = -mll(output, data.y.reshape(-1,))
            validation_loss += loss.item()
        validation_loss /= len(validation_loader.dataset)
    return validation_loss, output, data.y


def test(test_loader, device, model_gp, likelihood, mll):
    model_gp.eval()
    likelihood.eval()

    with torch.no_grad():
        test_loss = 0.0
        for data in test_loader:
            data = data.to(device)
            # output = likelihood(model_gp(data))
            output = likelihood(model_gp(data))

            loss = -mll(output, data.y.reshape(-1,))
            test_loss += loss.item()
        test_loss /= len(test_loader.dataset)
    return test_loss, output, data.y
