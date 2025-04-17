# %%
import numpy as np
from scipy.spatial.distance import cdist
from ase.io import read

# Define the file path
file_path = 'all_data/above_Al_O_down_C750_RC60-pos-1.xyz'

# Read the XYZ file
atoms = read(file_path, index='-1')

# Extract coordinates and element names
coordinates = atoms.get_positions()
element_names = atoms.get_chemical_symbols()

# Print the extracted coordinates and element names
print("Coordinates:")
print(coordinates)
print("\nElement Names:")
print(element_names)

# %%
cluster = coordinates[:-4]
adsorbate = coordinates[-4:]
# find pairwise distances between the cluster and the adsorbate
pairwise_distances = cdist(adsorbate, cluster)
print(np.min(pairwise_distances[-1, :]))
# %%
