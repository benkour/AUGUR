from scipy.optimize import linprog
import matplotlib.pyplot as plt
import random
import os
import numpy as np
import warnings
from scipy.stats import norm
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay, ConvexHull
from sklearn.metrics import pairwise_distances
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoLoader
import gpytorch
from helpers import get_graph
from models import BaseGNN, ExactGPModel
from placement import *
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp
import glob

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ignore the warning: Run   timeWarning: divide by zero encountered in true_divide
#   cm = (atomic_nums*atomic_nums.T) / pairwise_distances(pos)
warnings.filterwarnings("ignore", category=RuntimeWarning)
save_pos = True
with open('config.json', 'r') as json_file:
    config = json.load(json_file)

# Extract the NR_OF_ABSORBANTS value from the configuration
adsorbate_coordinates = read_last_snapshot('adsorbate.xyz')

# Extract the NR_OF_ABSORBANTS value from the configuration
NR_OF_ABSORBANTS = adsorbate_coordinates.shape[0]
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# %%


def generate_samples_in_batches(atoms, cluster, adsorbate_coordinates, min_distances, min_dist, device, indices, xsamples):
    results = []
    for i in indices:
        graph = generate_sample(
            atoms, cluster, adsorbate_coordinates, min_distances, min_dist, device, i)
        results.append((i, graph))
        output_dir = 'evaluating_positions_for_bo'
        file_path = os.path.join(output_dir, f'graph_{i}.pt')
        torch.save(graph, file_path)
        print("created sample: ", i)
    return results


def generate_sample(atoms, cluster, adsorbate_coordinates, min_distances, min_dist, device, index):
    Form_sample = place_adsorbate(
        atoms, cluster, adsorbate_coordinates, min_distances, min_dist)
    new_X = np.concatenate((cluster, Form_sample), axis=0)

    graph = get_graph(atoms, new_X)

    return index, graph.to(device)


def optimum_energy(model_gp, data_list):
    best_so_far = np.inf
    # shouldnt this give the target y? if we evaluate on the train data,
    # the posterior returns the mean as the exact target.

    means = -model_gp(Batch.from_data_list(data_list)
                      ).mean.cpu().detach().numpy()
    # uncertainty

    # TODO: depending on the target, we need to adjust this to max/min  (with log normalization should be max)
    best_so_far = means.max()
    return best_so_far


def points_on_sphere(center=None, N=100000, radius=1):
    '''generate N points on the surface of a sphere with given radius and center'''
    # spherical coordinates
    theta = np.random.uniform(0, 2*np.pi, (N,))
    phi = np.random.uniform(0, np.pi, (N,))
    # cartesian coordinates
    x = radius*np.sin(phi)*np.cos(theta)
    y = radius*np.sin(phi)*np.sin(theta)
    z = radius*np.cos(phi)
    points = center + np.vstack((x, y, z)).T
    return points
    # optimizer.zero_grad()
    # output from model


def is_feasible(points, hull):
    '''returns an array of boolean values that indicate which points are feasible and which are not.'''
    # if point is not in simplex
    return hull.find_simplex(points) < 0


def distance_to_hull(point, points):
    c = np.hstack((np.zeros(points.shape[1]), [-1]))
    A = np.vstack((np.hstack((points, np.ones((points.shape[0], 1)))), np.hstack(
        (point.reshape(1, -1), [[1]]))))
    b = np.ones(points.shape[0] + 1)
    b[-1] = -1
    lp = linprog(c, A_ub=A, b_ub=b)
    projection = lp.x[:-1]

    # Calculate the distance from the point to its projection
    distance = np.linalg.norm(point - projection)
    return distance


class BO:
    def __init__(self, adsorbate_coordinates, cluster, atoms, opt, likelihood, model_filenames,
                 method="ucb", tradeoff=0, sample_size=1, device='cpu'):
        self.cluster = cluster
        self.atoms = atoms
        self.opt = opt
        self.model_filenames = model_filenames
        self.method = method
        self.tradeoff = tradeoff
        self.sample_size = sample_size
        self.device = device
        self.likelihood = likelihood
        self.adsorbate_coordinates = adsorbate_coordinates
        self.std_flag = False

    def create_samples(self):
        output_dir = 'evaluating_positions_for_bo'

        # list of Data input objects to feed to GNNGP
        if config["create_new_samples"]:

            self.Xsamples = [None] * self.sample_size

            # NOTE: we could generate C depending on the convex hull of the adsorpant. Have to modify the is_feasible function and the generation of C points.
            # print(cluster)
            counter = 0

            # create dictionary of minimum allowed distances between atoms
            with open('min_distances.json', 'r') as json_file:
                min_distances = json.load(json_file)

            min_dist = np.zeros(
                (NR_OF_ABSORBANTS, len(self.atoms)-NR_OF_ABSORBANTS))

            for i in range(len(self.atoms) - NR_OF_ABSORBANTS):
                atom = self.atoms[i]
                for j in range(NR_OF_ABSORBANTS):
                    min_dist[j, i] = min_distances[self.atoms[len(
                        self.atoms) - NR_OF_ABSORBANTS + j]][atom]
            if config["parallel_sample_creation"]:
                # Specify the number of workers (cores) to use
                num_workers = config['number_of_cores']
                chunk_size = (self.sample_size + num_workers -
                              1) // num_workers  # Calculate chunk size

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [
                        executor.submit(generate_samples_in_batches, self.atoms, self.cluster, self.adsorbate_coordinates,
                                        min_distances, min_dist, self.device, range(i * chunk_size, min((i + 1) * chunk_size, self.sample_size)), self.Xsamples)
                        for i in range(num_workers)
                    ]

                    for future in as_completed(futures):
                        batch_results = future.result()
                        for index, graph in batch_results:
                            self.Xsamples[index] = graph[1]
            else:
                for counter in range(self.sample_size):
                    Form_sample = place_adsorbate(
                        self.atoms, self.cluster, self.adsorbate_coordinates, min_distances, min_dist)
                    # Form_samples.append(Form_sample)
                    new_X = np.concatenate(
                        (self.cluster, Form_sample), axis=0)
                    graph = get_graph(self.atoms, new_X)
                    file_path = os.path.join(output_dir, f'graph_{counter}.pt')
                    torch.save(graph, file_path)
                    self.Xsamples[counter] = graph
                    print("number of samples created: ", counter)

            print("samples created")
        else:

            # REad the samples from the directory.
            # read how many samples are in the directory
            directory = 'evaluating_positions_for_bo'
            file_ending = '*.pt'
            file_list = glob.glob(os.path.join(directory, file_ending))
            num_files = len(file_list)
            self.Xsamples = [None] * num_files
            for i, file_path in enumerate(file_list):
                graph = torch.load(file_path)
                self.Xsamples[i] = graph[1]

    def use_model(self):
        self.mu = 0
        self.std = 0
        for filename in self.model_filenames:
            self.mu_temp = []
            self.std_temp = []
            model = torch.load(os.path.join(
                os.getcwd(), filename), map_location=self.device, weights_only=False)
            model.device = self.device
            model = model.to(self.device)
            self.likelihood = self.likelihood.to(self.device)
            # print(model.device,self.device)
            test_loader = GeoLoader(self.Xsamples, batch_size=100)
            for test_data in test_loader:  # check layer x1 and x2 of the GP
                test_data = test_data.to(self.device)
                # consider uing likelihood)
                test_predictions = self.likelihood(model(test_data))
                # test_predictions = self.likelihood(model(test_data))# consider uing likelihood)

                # print("device " , test_predictions.mean.device, test_predictions.stddev.device)

                try:
                    self.mu_temp = np.hstack(
                        (self.mu_temp, test_predictions.mean.detach().cpu().numpy()))
                    self.std_temp = np.hstack(
                        (self.std_temp, test_predictions.stddev.cpu().detach().numpy()))
                except:
                    self.mu_temp = test_predictions.mean.cpu().detach().numpy()
                    self.std_temp = test_predictions.stddev.cpu().detach().numpy()
            self.mu_temp, self.std_temp = np.asarray(
                self.mu_temp), np.asarray(self.std_temp)
            self.std_temp = np.reshape(
                self.std_temp, (self.std_temp.shape[0], 1))

            self.mu_temp = np.reshape(self.mu_temp, (self.mu_temp.shape[0], 1))
            self.mu_temp = -self.mu_temp
            self.std_temp = np.sqrt(self.std_temp)

            # self.mu_temp.append(test_predictions.mean.cpu().detach().numpy())
            # self.std_temp.append(test_predictions.stddev.cpu().detach().numpy())   #
            # self.mu_temp.append(test_predictions.mean.cpu().detach().numpy())
            # self.std_temp.append(test_predictions.stddev.cpu().detach().numpy())   #
            if self.std_flag:

                try:
                    self.mu = np.hstack((self.mu, self.mu_temp))
                    self.std = np.hstack((self.std, self.std_temp))
                except:
                    self.mu = self.mu_temp
                    self.std = self.std_temp
            else:
                self.mu += self.mu_temp / len(self.model_filenames)
                self.std += self.std_temp / len(self.model_filenames)

        if self.std_flag:
            indices = np.argmin(self.std, axis=1)
            self.mu = self.mu[np.arange(self.mu.shape[0]), indices]
            self.std = self.std[np.arange(self.std.shape[0]), indices]

    def create_predictions(self):
        try:
            self.use_model()
        except:
            self.create_samples()
            self.use_model()

    def expected_improvement(self):
        with np.errstate(divide='warn'):
            imp = self.mu - self.opt - self.tradeoff
            Z = imp / self.std
            ei = imp * norm.cdf(Z) + self.std * norm.pdf(Z)
            ei[self.std == 0.0] = 0.0
        return ei

    def get_score(self):
        print("method: ", self.method, " tradeoff: ", self.tradeoff)
        if self.method == "pi":

            scores = norm.cdf((self.mu - self.opt) / (self.std+1E-9))

        elif self.method == "pe":
            scores = self.expected_improvement()

        elif self.method == "ucb":
            scores = self.mu + self.tradeoff*self.std

        # ix = np.argsort(scores, axis=0)[::-1][:1].reshape(-1)
        ix = np.argmax(scores)
        # print(ix)
        best_position = (
            self.Xsamples[ix].pos[-NR_OF_ABSORBANTS:])
        return ix, best_position

# %%
