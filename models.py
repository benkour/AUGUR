import torch
import gpytorch
import numpy as np
import random
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import NNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import TransformerConv, ChebConv, GATConv
# from torch_geometric.nn.glob import attention


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)


class BaseGNN(nn.Module):
    def __init__(self,
                 nodefeat_num=3, edgefeat_num=2,
                 nodeembed_to=3, edgeembed_to=2):
        super().__init__()
        # Embeddings
        self._node_embedding = nn.Sequential(
            nn.Linear(nodefeat_num, nodeembed_to), nn.ReLU())
        self._node_embednorm = nn.BatchNorm1d(nodeembed_to)
        self._edge_embedding = nn.Sequential(
            nn.Linear(edgefeat_num, edgeembed_to), nn.ReLU())
        self._edge_embednorm = nn.BatchNorm1d(edgeembed_to)

        # Graph Convolutions
        self._first_conv = NNConv(
            nodeembed_to,  # first, pass the initial size of the nodes
            nodeembed_to,  # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, nodeembed_to**2), nn.ReLU()
            ), aggr='mean'

        )
        self._first_conv_batchnorm = nn.BatchNorm1d(nodeembed_to)

        self._second_conv = NNConv(
            nodeembed_to,  # first, pass the initial size of the nodes
            nodeembed_to,  # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, nodeembed_to**2), nn.ReLU()
            ), aggr='mean'

        )
        self._third_conv_batchnorm = nn.BatchNorm1d(nodeembed_to)
        self._third_conv = NNConv(
            nodeembed_to,  # first, pass the initial size of the nodes
            nodeembed_to,  # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, nodeembed_to**2), nn.ReLU()
            ), aggr='mean'

        )
        self._second_conv_batchnorm = nn.BatchNorm1d(nodeembed_to)

        # # self._third_conv = ChebConv(in_channels=nodeembed_to, out_channels=nodeembed_to, K=2)
        self._third_conv_batchnorm = nn.BatchNorm1d(nodeembed_to)

        self._third_conv = GATConv(nodeembed_to, nodeembed_to, heads=1)

        # self._third_conv = NNConv(
        #     nodeembed_to, # first, pass the initial size of the nodes
        #     nodeembed_to, # and their output-size
        #     nn.Sequential(
        #         nn.Linear(edgeembed_to, nodeembed_to**2), nn.ReLU()
        #     ),aggr= 'mean'

        # )

        self._fourth_conv = NNConv(
            nodeembed_to,  # first, pass the initial size of the nodes
            nodeembed_to,  # and their output-size
            nn.Sequential(
                nn.Linear(edgeembed_to, nodeembed_to**2), nn.ReLU()
            ), aggr='mean'

        )
        self._fourth_conv_batchnorm = nn.BatchNorm1d(nodeembed_to)

        # Pooling and actuall prediction NN
        # takes batch.x and batch.batch as args
        self._pooling = [global_mean_pool, global_max_pool]
        # shape of one pooling output: [B,F], where B is batch size and F the number of node features.
        # shape of concatenated pooling outputs: [B, len(pooling)*F]
        self._predictor = nn.Sequential(
            nn.Linear(nodeembed_to*len(self._pooling), nodeembed_to),
            nn.ReLU(),
            nn.BatchNorm1d(nodeembed_to*3),
            nn.Linear(nodeembed_to*3, nodeembed_to*2),
            nn.ReLU(),
            nn.Linear(nodeembed_to*2, nodeembed_to),
            nn.ReLU(),
            nn.Linear(nodeembed_to, 6),
            nn.ReLU()

        )
        self._predictor.apply(init_weights)

    def forward(self, batch: Batch):
        node_features, edges, edge_features, mask, batch_vector = \
            batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.mask, batch.batch
        #### LNN#####
        # embed the features
        # node_features = self._node_embednorm(
        #     self._node_embedding(node_features))
        # edge_features = self._edge_embednorm(
        #     self._edge_embedding(edge_features))
        # edge_features = transformer(edge_features)
        # do graphs convolutions
        # node_features =self._first_conv_batchnorm(self._first_conv(
    # node_features, edges, edge_features))
        node_features = self._first_conv_batchnorm(self._first_conv(
            node_features, edges, edge_features))
        node_features = self._second_conv_batchnorm(self._second_conv(
            node_features, edges, edge_features))
        node_features = self._third_conv_batchnorm(
            self._third_conv(node_features, edges))
        node_features =self._fourth_conv_batchnorm(self._fourth_conv(
            node_features, edges, edge_features))
        # node_features = self._first_conv(node_features, edges, edge_features)
        # node_features = self._second_conv(node_features, edges, edge_features)

        # node_features = self._first_conv_batchnorm(self._first_conv(

        # now, do the pooling
        # pooled_graph_nodes = torch.cat([p(node_features[mask], batch_vector[mask]) for p in self._pooling], axis=1)
        pooled_graph_nodes = torch.cat(
            [p(node_features, batch_vector) for p in self._pooling], axis=1)

        # outputs = self._predictor(pooled_graph_nodes)
        return pooled_graph_nodes  # ready for a loss


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # +
        self.mean_module = gpytorch.means.ZeroMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=6))  # lengthscale for each dimension. Same shape as len(pooling_layer) # add white noise kernel, good for regression
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=12) # lengthscale for each dimension. Same shape as len(pooling_layer) # add white noise kernel, good for regression
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        # self.covar_module =  gpytorch.kernels.RBFKernel(ard_num_dims=6)
        # self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=6) #  + gpytorch.kernels.LinearKernel(num_dimensions=6)
        # self.covar_module =  gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=6))#+ gpytorch.kernels.LinearKernel()
        # self.covar_module = gpytorch.kernels.RBFKernel(
        #     ard_num_dims=6)  # + gpytorch.kernels.LinearKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=3, ard_num_dims=6))  # + gpytorch.kernels.LinearKernel()
        # self.covar_module = gpytorch.kernels.RFFKernel(num_samples = 5, ard_num_dims = 6) #+ gpytorch.kernels.LinearKernel()
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=6) #  + gpytorch.kernels.LinearKernel(num_dimensions=6)
        # self.covar_module = gpytorch.kernels.ArcKernel(self.base_kernel,
        #               angle_prior=gpytorch.priors.GammaPrior(0.5,1),
        #               radius_prior=gpytorch.priors.GammaPrior(3,2),
        #               ard_num_dims=6)

    def forward(self, x):
        mean_x = self.mean_module(x)  # .to(self.device)
        covar_x = self.covar_module(x)  # .to(self.device)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GNNGP(gpytorch.models.GP):

    def __init__(self, feature_extractor, gp, train_x, train_y, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device
        self.feature_extractor = feature_extractor
        self.gp = gp
        self.train_x = train_x  # must be a List of pyg Data
        self.train_y = train_y
        if self.training:
            train_x_features = self.feature_extractor(
                Batch.from_data_list(self.train_x))
            self.gp.set_train_data(inputs=train_x_features.to(
                device), targets=self.train_y, strict=False)  # consider strict=True
        # self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        # setting training features and labels
        if self.training:
            train_x_features = self.feature_extractor(x)
            # train_x_features = self.scale_to_bounds(train_x_features)

            self.gp.set_train_data(inputs=train_x_features.to(
                self.device), targets=self.train_y)  # consider strict=True
        if self.training:
            x1 = train_x_features
        else:
            # x1 = self.scale_to_bounds(self.feature_extractor(x)).to(device)
            x1 = self.feature_extractor(x).to(self.device)

        # actual forward
        # x1 = self.feature_extractor(x).to(device)
        x2 = self.gp(x1)
        return x2
