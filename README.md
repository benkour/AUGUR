# AUGUR

<p align="center">
  <img src="imgs/AUGUR.png"  width="200" />
    <figcaption style="font-size: 0.8em;">AUGUR (https://runeberg.org/nfbb/0226.html) Public Domain.</figcaption>
</p>

The code for the paper AUGUR, A flexible and efficient optimization algorithm for identification of optimal adsorption sites. We present the pipeline for the CHINI cluster to allow for a flexible template for ease of use.

**Requirements**
Our code has been tested on python 3.10.13 and ubuntu 22.04 (jammy) using visual code as the IDE. As a rule, everything should work in other systems as well but we cant guarantee it. The library versions we tested AUGUR with can be found in the requirements.txt. They can also be installed directly from requirements.txt using "pip install -r requirements.txt" from the folder level the requirements.txt resides.

**How to input your data**


  - place all your .xyz files (your so far simulation outputs) in the data_raw folder. (For example of output format have a look at the data_raw folder of the current repository) Minimum of 3 files is required before AUGUR can run
  - save the coordinates of your adsorbate in the adsorbate.xyz file in the source directory
  - Should you wish to place the adsorbate atoms at specific distances (for example O and Z should have a minimum 3.54 distance or something like this), complete the distances.json (you can see an example in the repo). Otherwise specify in config.json that you wish to use van der waals distances which will be automatically calculated
  
  
 The config.json is where all the model parameters are to be defined
 **config.json parameters**

  - "name_of_system" is a list of all the systems you wish to include. This is especially useful for when you have multiple nuclearities. Simply define a list of strings that are contained in the file names you wish to include.
   - "training_systems" a list of strings containing the systems you ll use for training AUGUR
   - "prediction_system" a single string for the system you wish to create the energy surface and determine the optimal adsorption site.
   - "energy_of_adsorbing" a list of the energies of the bare clusters/surfaces of the training systems.
   - energy_of_adsorbate" a value (in a list) of the energy of the bare adsorbate
   - "periodicity" true or false, depending on if your system is periodic or not
    - "cell_vectors" if your system is periodic define the cell vectors
   - "surface_cluster" define if the adsorbing molecule is a cluster or a surface by giving it 'cluster' or 'surface'
   - "van_der_waals_distances", true or false, depending on whether you want the adsorbate atoms to be placed within user defined distances (defined in distances.json) or by taking the sum of the van der waals distances of the elements
   - "specific_atom_closest" true or false depending on whether you want a specific adsorbate atom to always be placed nearest to the adsorbing molecule]
   - "which_atom" the index of the adsorbate atom you wish to place closer (simple numeric input, 0 for first , 1 for second, -1 for last etc)
   - "probability_of_closest" value between 0 and 1 giving the probability of placing the aforementioned atom closest (1 is always)
   - "non_viable_surface_atoms" list of element symbol strings Use in case you wish to exclude some adsorbing atoms from being considered (if you know a priori that they will not lead to favourable interactions)
   - "reprocess_data_for_training" true or false whether you wish to reprocess the data_raw before running the code or relying on previous processing (for example useful if you change processing systems or parameters)
   - cut_off_distance" simple number, the maximum distance that will generate an edge between two atoms]
   - "sample_size" number of samples to be evaluated by BO, simple number
   - "epochs" simple number denoting the number of training epochs
   - "create_new_samples" , true or false, whether you want to create new placement positions to be evaluated by BO or take the previous ones. Useful for when the cut_off_distance is different so sample graphs need to change. If everything is the same it is much more efficient to use already generated samples
   - parallel_sample_creation, true or false, denotes whether to create the samples for evaluation in parallel or not (makes the process much faster)
   -  "number_of_cores" , number, how many cores to be used for the parallel sample generation. Useful for benchmarking and hardware specific optimization
   - bo_round, number, denotes how many BO rounds have been performed so far (so the naming happens accordingly)
   - bo_acquisition, list of strings denoting which acquisition functions to use. So far supported , ucb (upper confidence bound), pe (expected improvement) , pi (probability of improvement)
   - bo_tradeoff a list of values denoting the tradeoff. a high value denotes exploration, a low exploitation. multiple values are supported
   - train, true or false, whether to retrain the model or not (useful for when you want to simply generate more points to evaluate existing models
   - optimize, true or false, whether to run bo or not, Useful for when you want to simply evaluate how the model parameters affect the accuracy
   - plot_flag, true or false, whether to plot results or not. If true the pipeline wont move till you close the graph)
    
    
**What to run**

once you configure everything run main.py


**Results**

If everything is done correctly, 
-the bo suggestions should appear in the bo_data folder. 
- the model pickles should appear in the model folder
- the energy surface, std surface and their corresponding csv containing the data should appear in the figures folder so long as the plot_flag in the config.json is set to true
- the  truth vs prediction graph and its corresponding csv containing the data should appear in the figures folder so long as the plot_flag in the config.json is set to true
- the proposed adsorption site visualization appear in the figures folder so long as the plot_flag in the config.json is set to true

**Examples**
Have a look at the branch silicene for periodic surface with complex adsorbate. Note that sometimes the parallel sample creation hangs. In this case resort to the sequential one
Have a look at the branch two_systems to see how to train on two systems and optimize one. NOTE: even if the adsorbate is the same for both systems you still need to specify the adsorbate energy twice, once for every system as seen in the example branch.
(Results may vary from the ones presented in the paper as we didnt recreate the exact architecture and other parameters for the sake of simplicity)


**Additional Information for advanced code alteration**
You can go into the model.py and change the model architecture at will. This is where both the GNN and GP are located. 

You can choose to include more or fewer points of the simulation trajectory by going into the create_data.py and adding more values in the index variable (should look like this index = [5,10,20, -1])
be wary! If you add too many points that are too similar to each other you will end up with a non invertible covariance matrix. So change these values keeping them sparse enough to be distinct but not too sparse so that information is not included. Trial and error might be required. (Also, dont put more points than the simulation has)


