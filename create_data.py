# %%
import re
import numpy as np
import torch
from helpers import *
import matplotlib.pyplot as plt
from pathlib import Path
import json

# %%


def normalization(a):
    return (a - a.min(axis=0))/(a.max(axis=0) - a.min(axis=0)), a.max(axis=0), a.min(axis=0)


def standardization(a):
    return (a - a.mean(axis=0))/a.std(axis=0), a.mean(axis=0), a.std(axis=0)


# adjust this list for types you want to process

raw_data_path = Path.cwd() / 'data_raw'
processed_data_path = Path.cwd() / 'data_processed'

# default
np.seterr(divide='ignore', invalid='ignore')

# clear processed data folder
for file in processed_data_path.glob('*'):
    file.unlink()

# get all files with extension in directory


def files_with_extension(directory, extension):
    return sorted(directory.rglob(f'*{extension}'))

# read files and process them


index = [0, 10, 20, -1]


def process_files(files, index=index):
    graph_list = []
    energies = []
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    types_to_process = config['name_of_system']
    # which iterations of the files to pick: we pick the last ones, since they have the lowest energy
    file_indices = {t: 0 for t in types_to_process}
    # read name_of_system from config.json

    # energy_adsorp_dict = {
    #     config['name_of_system']: (
    #         config['energy_of_adsorbing'], config['energy_of_adsorbate'])
    # }

    energies_of_adsorbing = config['energy_of_adsorbing']
    energies_of_adsorbate = config['energy_of_adsorbate']

    # Initialize the dictionary
    energy_adsorp_dict = {}

    # Populate the dictionary by iterating through the lists
    for name, energy_adsorbing, energy_adsorbate in zip(types_to_process, energies_of_adsorbing, energies_of_adsorbate):
        energy_adsorp_dict[name] = (energy_adsorbing, energy_adsorbate)

    for filename in files:
        print("Processing file: ", filename)

        # check filename for type
        type_key = next(
            (key for key in types_to_process if key.lower() in filename.stem.lower()), None)
        if not type_key:
            continue
        file_index = file_indices[type_key]
        isolated_en_cluster, isolated_en_mol = energy_adsorp_dict[type_key]
        e_adsorp = isolated_en_cluster + isolated_en_mol
        i = 0
        with open(filename, 'r') as file:
            energy_temp, graph_temp = [], []
            for line in file:
                if line.strip().isdigit():
                    atom_num = int(line.strip())
                    atom_count = 0
                    atoms, coordinates = [], []
                    i += 1
                    continue
                energy_match = re.search(r'E\s*=?\s*(-?\d+\.\d+)', line)
                if energy_match:
                    e_value = float(energy_match.group(1))
                    energy_temp.append(e_value)
                    continue

                if re.match(r'^\s*[A-Z][a-z]?', line):
                    atom, x1, x2, x3 = line.split()
                    atoms.append(atom)
                    coordinates.append([float(x1), float(x2), float(x3)])
                    atom_count += 1
                    if atom_count == atom_num:
                        graph_temp.append(get_graph(np.array(atoms), np.array(
                            coordinates), e_value, distance=12))
                        # from sklearn.metrics import pairwise_distances

                        # print("minimum distance = ",pairwise_distances(np.array(coordinates[:-1]))[:-1,-1].min())
            # some files contain many iterations, for these files we select the last iterations (i.e. index -1,-2, etc.)
            if i > 1:
                graph_temp = [graph_temp[ind] for ind in index]
                energy_temp = [energy_temp[ind] for ind in index]
            for i, graph in enumerate(graph_temp):
                graph_index = index[i] if i < len(index) else i
                graph_list.append((type_key, graph, file_index, graph_index))

            for energy in energy_temp:
                energies.append(energy - e_adsorp)
            save_atoms(processed_data_path, type_key + '_atoms.txt', atoms)
            file_indices[type_key] += 1
    return graph_list, energies


def create_data_function():
    # extract all .xyz files
    file_list = files_with_extension(raw_data_path, '.xyz')
    graph_list, energies = process_files(file_list, index)

    # normalize energies
    # plt.hist(energies)
    # plt.show()
    energies = np.array(energies)
    print(energies)

    mean, std = np.mean(energies), np.std(energies)
    # SAVE MEAN , STD and index in a txt called info.txt
    # index is index = [5, 10, 50, -1]
    with open('info.txt', 'w') as file:
        file.write(f"mean = {mean}\n")
        file.write(f"std = {std}\n")
        file.write(f"index = {index}\n")

    print("mean = ", mean, "std = ", std)
    energies, energie_max, energie_min = standardization(energies)

    # energies, energie_max, energie_min = normalization(energies)
    # plt.hist(energies)
    # plt.show()
    # print(energie_max, energie_min)
    # shuffle data
    for i, (type_key, graph, _, _) in enumerate(graph_list):

        graph.y = energies[i]

    processed_data_path.mkdir(parents=True, exist_ok=True)
    for type_key, graph, file_index, data_point_index in graph_list:
        filename = f'{type_key}_{file_index}_data_{data_point_index}.pt'
        torch.save(graph, processed_data_path / filename)
# %%
