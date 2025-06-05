# AUGUR: Optimal Adsorption Site Identification

<p align="center">
  <img src="imgs/AUGUR.png"  width="200" />
    <figcaption style="font-size: 0.8em;">AUGUR (https://runeberg.org/nfbb/0226.html) Public Domain.</figcaption>
</p>

This repository includes the code base of the paper **"AUGUR, A flexible and efficient optimization algorithm for identification of optimal adsorption sites"**. In here, we present the pipeline for the Chini clusters, silicene surface and training on two systems (all in separate branches) to allow for a flexible template for ease of use.

## Table of Contents

1. [Key Features (AUGUR) ](#key-features)
2. [Software versions tested ](#system-requirements)
4. [Parameter and Input Specification](#parameter-specification)
3. [Command Line Usage](#installation)
4. [Expected Output](#output)
5. [Tutorials](#tutorials)
6. [Advanced Usage and troubleshooting](#advanced)
7. [How to cite AUGUR](#citation)



## Key Features <a name="key-features"></a>

This section can include what makes AUGUR special in comparision to the existing methods.
- Stochastic GNN
- Flexible
- Symmetry invariant
- No hand crafted features needed
- Easy-to-Use

## System Requirements <a name="system-requirements"></a>

**Tested Configuration**:

- Operating System: Ubuntu 22.04 (jammy)
- IDE: Visual Studio Code 

**Note**: As a rule, everything should work on other systems as well but we cant guarantee it.

**Tested quantum chemistry packages**:
- ORCA 5.04 (Trajectory file compatibility for systems: chini clusters)
- CP2k (systems - Silicene)

*For using AUGUR with any other simulation softwares, adjust the create_data.py to read the respective trajectory formats or contact us at Johnkouroudis[at]gmail[dot]com. (change the things in [ ] with their symbols. They are only there to confuse the vulture bots prowling the internet *


## Parameter and Input Specification<a name="parameter-specification"></a>


  - Place all the optimized .xyz trajectory files obtained from the converged simulations in the **data_raw** folder. (For example, for the current software in use (Orca 5.04), the optimized trajectory files look as in the data_raw folder of the current repository). Note that a minimum of 3 converged trajectory files are required before AUGUR can run.
  - save the coordinates of your adsorbate in the adsorbate.xyz file in the source directory
  - Should you wish to place the adsorbate atoms at specific distances (for example O and Zn should have a minimum 3.54 distance or something like this), complete the distances.json (you can see an example in the repo). Otherwise specify in config.json that you wish to use van der waals distances which will be automatically calculated

  will be automatically calculated



 The config.json is where all the model parameters are to be defined

 **config.json parameters**
| Parameter                     | Type    | Description                                                                                                                                                                                                                                                                                |
| ----------------------------- | ------  | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `name_of_system`              | List    | is a list of all the systems you wish to include. This is especially useful for when you have multiple nuclearities. Simply define a list of strings that are contained in the file names you wish to include.                                                                             |
| `training_systems`            | List    | a list of strings containing the systems you ll use for training AUGUR                                                                                                                                                                                                                     |
| `prediction_system`           | String  | a single string for the system you wish to create the energy surface and determine the optimal adsorption site.                                                                                                                                                                            |
| `energy_of_adsorbing`         | List    | a list of the energies of the bare clusters/surfaces of the training systems.                                                                                                                                                                                                              |
| `energy_of_adsorbate`         | List    | a value (in a list) of the energy of the bare adsorbate                                                                                                                                                                                                                                    |
| `periodicity`                 | Bool    | true or false, depending on if your system is periodic or not                                                                                                                                                                                                                              |
| `cell_vectors`                | List    | if your system is periodic define the cell vectors                                                                                                                                                                                                                                         |
| `surface_cluster`             | String  | define if the adsorbing molecule is a cluster or a surface by giving it 'cluster' or 'surface'                                                                                                                                                                                             |
| `van_der_waals_distances`     | Bool    | true or false, depending on whether you want the adsorbate atoms to be placed within user defined distances (defined in distances.json) or by taking the sum of the van der waals distances of the elements                                                                                |
| `specific_atom_closest`       | Bool    | true or false depending on whether you want a specific adsorbate atom to always be placed nearest to the adsorbing molecule                                                                                                                                                               |
| `which_atom`                  | Int     | the index of the adsorbate atom you wish to place closer (simple numeric input, 0 for first , 1 for second, -1 for last etc)                                                                                                                                                               |
| `probability_of_closest`      | Float   | value between 0 and 1 giving the probability of placing the aforementioned atom closest (1 is always)                                                                                                                                                                                      |
| `non_viable_surface_atoms`    | List    | list of element symbol strings Use in case you wish to exclude some adsorbing atoms from being considered (if you know a priori that they will not lead to favourable interactions)                                                                                                        |
| `reprocess_data_for_training` | Bool    | true or false whether you wish to reprocess the data_raw before running the code or relying on previous processing (for example useful if you change processing systems or parameters)                                                                                                     |
| `cut_off_distance`            | Float?  | simple number, the maximum distance that will generate an edge between two atoms                                                                                                                                                                                                          |
| `sample_size`                 | Int     | number of samples to be evaluated by BO, simple number                                                                                                                                                                                                                                     |
| `epochs`                      | Int     | simple number denoting the number of training epochs                                                                                                                                                                                                                                       |
| `create_new_samples`          | Bool    | true or false, whether you want to create new placement positions to be evaluated by BO or take the previous ones. Useful for when the cut_off_distance is different so sample graphs need to change. If everything is the same it is much more efficient to use already generated samples |
| `parallel_sample_creation`    | Bool    | true or false, denotes whether to create the samples for evaluation in parallel or not (makes the process much faster)                                                                                                                                                                     |
| `number_of_cores`             | Int     | number, how many cores to be used for the parallel sample generation. Useful for benchmarking and hardware specific optimization                                                                                                                                                           |
| `bo_round`                    | Int     | number denotes how many BO rounds have been performed so far (so the naming happens accordingly)                                                                                                                                                                                           |
| `bo_acquisition`              | List    | list of strings denoting which acquisition functions to use. So far supported , ucb (upper confidence bound), pe (expected improvement) , pi (probability of improvement)                                                                                                                  |
| `bo_tradeoff`                 | List    | a list of values denoting the tradeoff. a high value denotes exploration, a low exploitation. multiple values are supported                                                                                                                                                                |
| `train`                       | Bool    | true or false, whether to retrain the model or not (useful for when you want to simply generate more points to evaluate existing models                                                                                                                                                    |
| `optimize`                    | Bool    | true or false, whether to run bo or not, Useful for when you want to simply evaluate how the model parameters affect the accuracy                                                                                                                                                          |
| `plot_flag`                   | Bool    | true or false, whether to plot results or not. If true the pipeline wont move till you close the graph)                                                                                                                                                                                    |

## Command-Line-Usage <a name="installation"></a>


- The specific library versions we tested AUGUR with can be found in the requirements.txt. They can be installed directly using `pip install -r requirements.txt` from the folder level the requirements.txt resides.

  ```
  # Clone the Repository
  git clone https://github.com/benkour/AUGUR.git
  # Make a virtual env
  python -m venv augur_env
  source ./augur_env/bin/activate
  pip install -r requirements.txt
  cd AUGUR
  ```
The code will also benefit from CUDA should you have an NVIDIA graphics card

**What to run** - once you configure everything run main.py


## Expected Output <a name="output"></a>

If everything is done correctly (and you have set the correct flags in the config.json to true),

- the bo suggestions should appear in the bo_data folder.
- the model pickles should appear in the model folder
- the energy surface, std surface and their corresponding csv containing the data should appear in the figures folder so long as the plot_flag in the config.json is set to true
- the  truth vs prediction graph and its corresponding csv containing the data should appear in the figures folder so long as the plot_flag in the config.json is set to true
- the proposed adsorption site visualization appear in the figures folder so long as the plot_flag in the config.json is set to true

## Tutorials <a name="tutorials"></a>

- the branch silicene contains an example for a for periodic surface (silicene) with complex adsorbate. Note that sometimes the parallel sample creation hangs. In this case resort to the sequential one
- the branch two_systems contains an example about how to train on two systems (Pt3 and Pt6) and optimize one (Pt6). NOTE: even if the adsorbate is the same for both systems you still need to specify the adsorbate energy twice, once for every system as seen in the example branch.
(Results may vary from the ones presented in the paper as we didnt recreate the exact architecture and other parameters for the sake of simplicity)


## Advanced Usage and troubleshooting <a name="advanced"></a>

- You can go into the model.py and change the model architecture at will. This is where both the GNN and GP are located.
- You can choose to include more or fewer points of the simulation trajectory by going into the create_data.py and adding more values in the index variable (should look like this index = [5,10,20, -1])
be wary! If you add too many points that are too similar to each other you will end up with a non invertible covariance matrix. So change these values keeping them sparse enough to be distinct but not too sparse so that information is not included. Trial and error might be required. (Also, dont put more points than the simulation has)
- Sometimes the parallel sample creation runs out of memory and hangs. In this case, you are adviced to run the data preprocessing first (changing the  reprocess_data_for_training variable to true and create_new_samples false in the configjson), running the code and then, run it again by setting the reprocess_data_for_training to false and create_new_samples to true. Sequential placement should always work.  

## How to cite AUGUR <a name="citation"></a>
if you found AUGUR useful you can cite it as follows
```
@article{kouroudis2025augur,
  title={AUGUR, a flexible and efficient optimization algorithm for identification of optimal adsorption sites},
  author={Kouroudis, Ioannis and Misciasci, Neel and Mayr, Felix and M{\"u}ller, Leon and Gu, Zhaosu and Gagliardi, Alessio and others},
  journal={npj Computational Materials},
  volume={11},
  number={1},
  pages={1--13},
  year={2025},
  publisher={Nature Publishing Group}
}
```

