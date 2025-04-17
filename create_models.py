# %%
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from ase.visualize import view
from ase.io import read, write
from mpl_toolkits.mplot3d import Axes3D
from ase import Atoms
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import pandas as pd
import copy
import time
import random
from training import *
from bo import *
from models import *
import os
import numpy as np
import warnings
from sklearn.utils import shuffle
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoLoader
import gpytorch
from helpers import *
from models import BaseGNN, ExactGPModel
from collections import Counter

seed = 42
# %%


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def create_model_function():
    # read the DATA_IDX from the info.txt as the array
    with open('info.txt', 'r') as file:
        for line in file:
            if 'index' in line:
                DATA_IDX = eval(line.split('=')[1])

    # DATA_IDX = [5, 10, 50, -1]

    set_seed(seed)
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    adsorbate_coordinates = read_last_snapshot('adsorbate.xyz')

    # Extract the NR_OF_ABSORBANTS value from the configuration
    NR_OF_ABSORBANTS = adsorbate_coordinates.shape[0]
    # %%
    BO_ITERATION = str(config["bo_round"])
    learning_rate = 1e-3
    EPOCHS = config['epochs']
    SAMPLE_SIZE = config['sample_size']
    Training_Type = config['training_systems']
    type_of_graph = "SPARSE"
    Prediction_Type = config['prediction_system']
    # test_set = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # %%
    data_list, data_dict = data_preparation(types_to_include=[Prediction_Type])
    atoms = atom_extraction(Prediction_Type)
    # %%
    cluster = data_dict[Prediction_Type][0][DATA_IDX[0]
                                            ][0].pos[:-NR_OF_ABSORBANTS]
    cluster_origin = list(zip(atoms[:-NR_OF_ABSORBANTS], cluster))

    # data_list,data_dict = data_preparation(types_to_include = Training_Type)
    # atoms = atom_extraction(Prediction_Type)
    # cluster = data_dict[Prediction_Type][0][0][0].pos[:-NR_OF_ABSORBANTS]
    # cluster_origin = list(zip(atoms[:-NR_OF_ABSORBANTS],cluster))
    # print(data_dict)
    # %%

    # %%

    # %% cross validation
    loss_history = []
    eval_loss_history = []

    best_model = None
    validation_prediction = []
    validation_std = []
    validation_truth = []
    model_filenames = []
    mean_temp = 0
    std_temp = 0
    truth_temp = 0
    counter_of_cross_validation = 0
    os.system(r'rm -r models/*.pth')
    os.system(r'rm -r bo_data/*.txt')
    all_data_combinations = [(key, file_idx, data_idx)
                             for key in data_dict
                             for file_idx in data_dict[key]
                             for data_idx in DATA_IDX]
    print("Training started")
    for type_key in data_dict:
        all_file_indices = list(data_dict[type_key].keys())

        # for validation_file_index in all_file_indices:
        for validation_file_index in np.random.choice(all_file_indices, 3, replace=False):

            # Split data into training, validation and test
            valid_exclusion = [(type_key, validation_file_index, data_idx)
                               for data_idx in DATA_IDX]
            available_data = [
                item for item in all_data_combinations if item not in valid_exclusion]

            validation_data = []
            for data_idx in DATA_IDX:
                validation_data.extend(
                    data_dict[type_key][validation_file_index][data_idx])

            test_choice = random.choice(
                [item for item in available_data if item[0] != type_key or item[1] != validation_file_index])
            test_data = []
            for data_idx in DATA_IDX:
                test_data.extend(data_dict[test_choice[0]]
                                 [test_choice[1]][data_idx])
            test_exclusion = [(test_choice[0], test_choice[1], data_idx)
                              for data_idx in DATA_IDX]

            training_data = []
            for item in available_data:
                if item not in test_exclusion:
                    training_data.extend(data_dict[item[0]][item[1]][item[2]])

            # To Tensor
            targets = torch.tensor(
                [data.y for data in training_data], dtype=torch.float).to(device)

            # Create DataLoader
            train_loader = GeoLoader(
                training_data, batch_size=len(training_data), shuffle=True)
            validation_loader = GeoLoader(
                validation_data, batch_size=len(validation_data), shuffle=False)
            test_loader = GeoLoader(
                test_data, batch_size=len(test_data), shuffle=False)

            noises = torch.ones(len(train_loader)) * 0.1
            likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                noise=noises, learn_additional_noise=True)

            # likelihood = gpytorch.likelihoods.GaussianLikelihood()

            feature_extractor = BaseGNN(
                data_list[0].x.shape[-1], data_list[0].edge_attr.shape[-1], data_list[0].x.shape[-1], data_list[0].edge_attr.shape[-1])
            gp = ExactGPModel(train_x=None, train_y=None,
                              likelihood=likelihood)
            feature_extractor.to(device)
            gp.to(device)
            model_gp = GNNGP(feature_extractor=feature_extractor, gp=gp,
                             train_x=training_data, train_y=targets, device=device)
            model_gp.to(device)
            optimizer = torch.optim.Adagrad(
                model_gp.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=0.8, patience=3,
                                                                   min_lr=0.0000001)

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                likelihood, model_gp)
            # print(file_index)
            best_validation_loss = float('inf')
            best_epoch = 0
            for epoch in range(1, EPOCHS+1):
                model_gp.train()
                likelihood.train()
                loss = train(train_loader, optimizer, device, model_gp, mll)

                scheduler.step(loss)

                means = model_gp(Batch.from_data_list(training_data)
                                 ).mean.detach().cpu().clone().numpy()
                # print(means, "means")
                model_gp.eval()
                likelihood.eval()
                std_print = model_gp(Batch.from_data_list(
                    training_data)).stddev.detach().cpu().clone().numpy()
                validation_loss, validation_output, validation_truth_temp = validate(
                    validation_loader, device, model_gp, likelihood, mll)
                if validation_loss < best_validation_loss:
                    _, test_output, test_truth = test(
                        test_loader, device, model_gp, likelihood, mll)
                    best_validation_loss = validation_loss
                    best_model = model_gp
                    best_epoch = epoch
                    mean_temp = test_output.mean.detach().cpu().clone().numpy()
                    std_temp = test_output.stddev.detach().cpu().clone().numpy()
                    validation_truth_temp_detached = test_truth.detach().cpu().clone().numpy()
                eval_loss_history.append({"validation": validation_loss})
            print(best_epoch, counter_of_cross_validation,
                  mean_temp, validation_truth_temp_detached)

            model_filename = 'models/best_model_' + Prediction_Type + \
                str(type_key)+str(type_key)+str(validation_file_index)+'.pth'
            model_filenames.append(model_filename)
            torch.save(best_model, os.path.join(os.getcwd(), model_filename))
            loss_history.append({"train": loss})
            validation_prediction.append(mean_temp)
            validation_std.append(std_temp)
            validation_truth.append(validation_truth_temp_detached)
    # %%

    # %%
    validation_prediction = array_reshape(validation_prediction)
    validation_std = array_reshape(validation_std)
    validation_truth = array_reshape(validation_truth)
    # %%
    name_of_data = 'error_mse'
    average_error = ((abs(validation_truth - validation_prediction)**2).mean())
    print(f'Average MSE: {average_error:.2f}')

    x_min = np.minimum(validation_prediction, validation_truth).min() - 0.1
    x_max = np.maximum(validation_prediction, validation_truth).max() + 0.1
    x = np.linspace(x_min, x_max, 400).squeeze()
    y = x
    if config["plot_flag"]:
        plt.figure(figsize=(10, 10))
        plt.plot(validation_prediction, validation_truth, 'o', c='#1f77b4')
        plt.plot(x, y, linestyle='dotted')

        plt.xlabel('Energy prediction (standardized)')
        plt.ylabel('True Energy (standardized)')
        plt.xlim(x_min, x_max)
        plt.ylim(x_min, x_max)
        plt.legend(title=f'Average MSE: {average_error:.2f}')

        plt.savefig("results/mse_"+'.pdf')
        plt.show()

    # %%
    # Create a DataFrame
    df = pd.DataFrame({
        'validation_truth': validation_truth.squeeze(-1),
        'validation_prediction': validation_prediction.squeeze(-1),
        'validation_std': validation_std.squeeze(-1)
    })

    # Save the DataFrame to a text file
    df.to_csv('results/'+name_of_data+'.txt', sep='\t', index=False)
