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


# %%


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# read the DATA_IDX from the info.txt as the array
with open('info.txt', 'r') as file:
    for line in file:
        if 'index' in line:
            DATA_IDX = eval(line.split('=')[1])

# DATA_IDX = [5, 10, 50, -1]


def bo_optimization():
    seed = 42
    set_seed(seed)
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    adsorbate_coordinates = read_last_snapshot('adsorbate.xyz')

    # Extract the NR_OF_ABSORBANTS value from the configuration
    NR_OF_ABSORBANTS = adsorbate_coordinates.shape[0]
    BO_ITERATION = str(config["bo_round"])
    learning_rate = 1e-3
    EPOCHS = config['epochs']
    SAMPLE_SIZE = config['sample_size']
    Training_Type = config['training_systems']
    type_of_graph = "SPARSE"
    Prediction_Type = config['prediction_system']
    test_set = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_list, data_dict = data_preparation(types_to_include=[Prediction_Type])
    atoms = atom_extraction(Prediction_Type)
    cluster = data_dict[Prediction_Type][0][DATA_IDX[0]
                                            ][0].pos[:-NR_OF_ABSORBANTS]
    cluster_origin = list(zip(atoms[:-NR_OF_ABSORBANTS], cluster))

    seed = 42

    # get the names of all files in the models folder
    model_filenames = ["models/"+f for f in os.listdir(
        'models') if f.endswith('.pth') and 'checkpoint' not in f]
    opt = 0
    avg_state_dict = {}
    for filename in model_filenames:
        model = torch.load(os.path.join(
            os.getcwd(), filename), weights_only=False)
        opt += optimum_energy(model, data_list)
    opt = opt/len(model_filenames)


    start_time = time.time()
    model_gp = torch.load(os.path.join(
        os.getcwd(), filename), weights_only=False)
    opt = optimum_energy(model_gp, data_list)
    print("inside bo_optimization")

    #
    # read the coordinates from the xyz file adsorbate.xyz

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    
    if config["van_der_waals_distances"]:
        surface_elements = atoms[:-NR_OF_ABSORBANTS]
        adsorbate_elements = atoms[-NR_OF_ABSORBANTS:]

        # Count the frequency of each element
        surface_counts = Counter(surface_elements)
        adsorbate_counts = Counter(adsorbate_elements)

        # Get unique elements and sort them by frequency
        unique_surface_elements = sorted(
            surface_counts.keys(), key=lambda x: surface_counts[x], reverse=True)
        unique_adsorbate_elements = sorted(
            adsorbate_counts.keys(), key=lambda x: adsorbate_counts[x], reverse=True)

        # Create the min_distances dictionary
        min_distances = {}
        for adsorbate_element in unique_adsorbate_elements:
            min_distances[adsorbate_element] = {}
            for surface_element in unique_surface_elements:
                vdw_radius_adsorbate = get_vdw_radius(adsorbate_element)
                vdw_radius_surface = get_vdw_radius(surface_element)
                min_distances[adsorbate_element][surface_element] = vdw_radius_adsorbate + \
                    vdw_radius_surface

        # Save the min_distances dictionary to a JSON file
        with open('min_distances.json', 'w') as json_file:
            json.dump(min_distances, json_file, indent=4)
    bo = BO(adsorbate_coordinates, cluster, atoms, opt, likelihood, model_filenames, method="pe",
            tradeoff=0, sample_size=SAMPLE_SIZE, device=device)
    print("inside bo_optimization")


    bo.method = 'pi'
    bo.std_flag = False
    # time how long it takes to create the samples and predictions
    start_time = time.time()
    print("before bo_optimization")

    bo.create_samples()
    print("after create samples")
    end_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))
    bo.create_predictions()
    # %%
    name = 'bo_data/' + config['prediction_system'] + \
        "_BO_iteration_"+BO_ITERATION
    nr = float(BO_ITERATION)

    # %%

    # %%

    for i in config["bo_acquisition"]:
        for j in config["bo_tradeoff"]:
            bo.method = i
            bo.tradeoff = j
            pos = bo.get_score()
            save_points(name+'_'+str(bo.tradeoff)+'_'+str(bo.method),
                        pos[1], cluster_origin, atoms[-NR_OF_ABSORBANTS:], nr)

    # plt.scatter(bo.Xsamples[:].pos[:,0], bo.Xsamples[:].pos[:,1],bo.Xsamples[:].pos[:,2], s=200, c=bo.std, cmap='gray')
    # %%
    positions = []
    for i in range(len(bo.Xsamples)):
        positions.append(bo.Xsamples[i].pos[-1, :])

    positions = np.array(positions)
    # read the unnormalization_mean and unnormalization_std from the info.txt
    # %%
    with open('info.txt', 'r') as file:
        for line in file:
            if 'mean' in line:
                unnormalization_mean = float(line.split('=')[1])
            if 'std' in line:
                unnormalization_std = float(line.split('=')[1])

    # %%
    if config["plot_flag"]:
        fig = plt.figure(figsize=(10, 10))
        axs = fig.add_subplot(projection='3d')
        p = axs.scatter(
            cluster[:, 0], cluster[:, 1], cluster[:, 2], s=100, alpha=1)
        # p =  axs.scatter(positions[:,0], positions[:,1], positions[:,2],c = (-bo.mu[:]*0.12309146 ) ,s = 50)
        # p =  axs.scatter(positions[:,0], positions[:,1], positions[:,2],c = bo.mu[:]    ,s = 50)
        p = axs.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        c=((-bo.mu[:]*unnormalization_std + unnormalization_mean)), s=50)

        # p = axs.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
        # c=((-bo.mu[:] + 275.32623719727127)/(-275.32623719727127 + 271.6193179728926)), s=50)
        # p = axs.scatter(positions[:, 0], positions[:, 1],
        #                 positions[:, 2], c=bo.std[:], s=50)
        axs.set_xlabel('X $[\AA]$', fontsize=20)
        axs.set_ylabel('Y $[\AA]$', fontsize=20)
        axs.set_zlabel('Z $[\AA]$', fontsize=20)

        cbar = fig.colorbar(p, ax=axs)
        cbar.ax.tick_params(labelsize=20)
        # , Learning and individual differences 103, 102274fontsize=20)
        cbar.set_label('Energy [eV]', fontsize=20)
        axs.set_title("Energy surface")
        plt.savefig('results/energy_surface.pdf')

        plt.show()

        fig = plt.figure(figsize=(10, 10))
        axs = fig.add_subplot(projection='3d')
        p = axs.scatter(
            cluster[:, 0], cluster[:, 1], cluster[:, 2], s=100, alpha=1)
        # p =  axs.scatter(positions[:,0], positions[:,1], positions[:,2],c = (-bo.mu[:]*0.12309146 ) ,s = 50)
        # p =  axs.scatter(positions[:,0], positions[:,1], positions[:,2],c = bo.mu[:]    ,s = 50)
        p = axs.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                        c=(bo.std[:]), s=50)

        axs.set_xlabel('X $[\AA]$', fontsize=20)
        axs.set_ylabel('Y $[\AA]$', fontsize=20)
        axs.set_zlabel('Z $[\AA]$', fontsize=20)

        cbar = fig.colorbar(p, ax=axs)
        cbar.ax.tick_params(labelsize=20)
        # , Learning and individual differences 103, 102274fontsize=20)
        cbar.set_label('Std', fontsize=20)
        axs.set_title("Uncertainty of energy surface")
        plt.savefig('results/std_surface.pdf')
        plt.show()

        # Create a DataFrame
    df = pd.DataFrame({
        'position_x': positions[:, 0],
        'position_y': positions[:, 1],
        'position_z': positions[:, 2],
        'bo_mu': ((-bo.mu[:]*unnormalization_std + unnormalization_mean)).squeeze(),
        'bo_std': bo.std[:].squeeze()
    })

    # Save the DataFrame to a CSV file
    df.to_csv("results/energy_surface_of_" +
              config["prediction_system"]+'.csv', index=False)
    # %%
    # %%
    bo.get_score()[1]
    # %%

    # open a random xyz from data_bo folder and visualize hte cluster
    # Step 1: List all .txt files in the data_bo folder

    # Step 2: Select a random file from the list
    if config["plot_flag"]:
        data_bo_folder = 'bo_data'
        txt_files = [f for f in os.listdir(
            data_bo_folder) if f.endswith('.txt')]
        for i in range(len(txt_files)):
            random_file = txt_files[i]
            file_path = os.path.join(data_bo_folder, random_file)

            # Step 3: Read the selected .txt file
            symbols = []
            positions = []

            with open(file_path, 'r') as file:
                for line in file:
                    parts = line.split()
                    if len(parts) == 4:
                        symbol, x, y, z = parts
                        symbols.append(symbol)
                        positions.append([float(x), float(y), float(z)])

            atoms = Atoms(symbols=symbols, positions=positions)

            # Step 4: Visualize the cluster with atoafter index = ms colored according to their type
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Define colors for different atom types
            colors = {
                'H': 'yellow',
                'Pt': 'silver',
                'C': 'grey',
                'O': 'red',
                'Si': 'cyan',
                'Al': 'cyan',
                'N': 'black',
                'B': 'cyan',
                'P': 'cyan',
                'Zn': 'blue',
                'F': 'blue',

                # Add more atom types and their colors as needed
            }

            for atom in atoms:
                ax.scatter(atom.position[0], atom.position[1], atom.position[2], color=colors.get(
                    atom.symbol, 'gray'), s=100)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(random_file)
            plt.savefig('results/proposed_position_'+str(random_file)+'.pdf')
            # SET axis limits to be the same for all three axes
            plt.show()
