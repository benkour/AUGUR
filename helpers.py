from ase.io import read
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from torch_geometric.data import Data, Batch
from ase.geometry import find_mic
from ase.cell import Cell
import json
import os
from scipy.spatial import Delaunay, ConvexHull
from ase.data import vdw_radii, chemical_symbols

# Function to get the van der Waals radius of an element


def get_vdw_radius(element):
    # Convert element symbol to atomic number
    atomic_number = chemical_symbols.index(element)
    return vdw_radii[atomic_number]


def cluster_convex_hull(points, cluster):
    hull = Delaunay(cluster)
    '''returns an array of boolean values that indicate which points are feasible and which are not.'''
    # if point is not in simplex
    return np.all(hull.find_simplex(points) < 0)


def get_periodic_distance(coordinates):

    # get cell vectors from config.json
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    # Extract the cell vectors
    cell_vectors = config['cell_vectors']
    a, b, c = np.array(cell_vectors[0]), np.array(
        cell_vectors[1]), np.array(cell_vectors[2])

    # get box size from coordinates
    num_points = len(coordinates)
    # Calculate the pairwise distance of every point to every point using vectorized operations
    cell = Cell([a, b, c])
    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    dist_vec, pairwise_distances = find_mic(diff.reshape(-1, 3), cell)
    pairwise_distances = pairwise_distances.reshape(num_points, num_points)
    return pairwise_distances


def get_graph(atoms, pos, target=None, distance=12):
    '''
    Get a Data graph from the numpy coordinates, the type of atom and the target.
    '''
    # edge index
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)
    a = np.arange(len(atoms))
    edges = np.array(np.meshgrid(a, a)).T.reshape(-1, 2).T
    edges = torch.tensor(edges, dtype=torch.int64)
    # read the atom to properties dictionary
    with open('element_properties.json', 'r') as json_file:
        data = json.load(json_file)
        atom_to_num = data['atom_to_num']
        atom_to_en = data['atom_to_en']
        atom_to_r = data['atom_to_r']

    atomic_nums = np.asarray([atom_to_num[atom] for atom in atoms])[
        :, np.newaxis]  # keep as numpy for later use
    electroneg = torch.tensor(np.asarray([atom_to_en[atom] for atom in atoms])[
                              :, np.newaxis], dtype=torch.float)
    atomic_radius = torch.tensor(np.asarray([atom_to_r[atom] for atom in atoms])[
                                 :, np.newaxis], dtype=torch.float)

    # In the loop we extract the nodes' embeddings, edges connectivity
    # and label for a graph, process the information and put it in a Data
    # object, then we add the object to a list

    # Node features
    # atomic number abd electronegativity # TODO: add atomic radius

    # Edge features
    # shape [N', D'] N': number of edges, D': number of edge features
    # cm matrix and bond matrix
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    periodicity = config['periodicity']
    if periodicity:
        pair_dist = get_periodic_distance(pos)
    else:
        pair_dist = pairwise_distances(pos)
    cm = (atomic_nums*atomic_nums.T) / pair_dist
    np.fill_diagonal(cm, 0.5*atomic_nums**2.4)

    mask = pair_dist < config['cut_off_distance']

    # Get the indices where the mask is True
    edge_indices = np.where(mask)

    # Extract the corresponding values
    edge_1, edge_2 = edge_indices
    cm_sparse = cm[mask]
    dist_sparse = pair_dist[mask]

    # Convert to numpy arrays
    cm_sparse = np.array(cm_sparse)
    dist_sparse = np.array(dist_sparse)

    # Convert to torch tensors
    cm_sparse = torch.tensor(cm_sparse[:, np.newaxis], dtype=torch.float)
    dist_sparse = torch.tensor(dist_sparse[:, np.newaxis], dtype=torch.float)

    edge_attr = torch.cat([cm_sparse, dist_sparse], dim=1)
    edges = torch.tensor(np.array([edge_1, edge_2]), dtype=torch.int64)
    if target:
        target = torch.tensor(target, dtype=torch.float)

    node_attrs = torch.cat(
        [torch.tensor(atomic_nums, dtype=torch.float), electroneg, atomic_radius], dim=1)
    distances_co = pair_dist[-1:, :]
    nearby_nodes = np.argwhere(distances_co < distance)[:, 1]
    nearby_nodes = np.array(list(set(nearby_nodes)))
    mask = torch.zeros((pair_dist.shape[0]), dtype=torch.bool)
    mask[nearby_nodes] = 1

    graph = Data(x=node_attrs,
                 pos=pos,
                 edge_index=edges,
                 edge_attr=edge_attr,
                 y=target,
                 mask=mask)

    return graph


def save_points(name, positions, cluster_origin, adsorbate_atoms, nr):
    positions = np.array(positions)
    positions = np.reshape(positions, (len(adsorbate_atoms), 3))
    with open(name+'.txt', "w") as file:
        file.write(f" {positions.shape[0]+len(cluster_origin)}\n")
        file.write(f"Simulation {nr}\n")
        for atom, coordinate in cluster_origin:
            line = f"{atom} {coordinate[0]} {coordinate[1]} {coordinate[2]}\n"
            file.write(line)
        for i in range(len(adsorbate_atoms)):
            line = f"{adsorbate_atoms[i]} {positions[i][0]} {positions[i][1]} {positions[i][2]}\n"
            file.write(line)
        file.write("\n######################################\n")


def save_atoms(root_path, txt_name, atoms):
    file_path = os.path.join(root_path, txt_name)
    if not os.path.isfile(file_path):
        np.savetxt(file_path, atoms, fmt='%s')
        print(f'File {txt_name} saved.')
    else:
        print(f'File {txt_name} exists')


def array_reshape(a):
    a = np.array(a)
    a = np.reshape(a, (a.shape[0]*a.shape[1], 1))
    return a


def read_last_snapshot(filename):
    atoms = read(filename, index='-1')
    coordinates = atoms.get_positions()
    return coordinates


# %%
