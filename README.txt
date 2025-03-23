==================================================
README - DeepLearningProject_armftn
==================================================

This Deep Learning project presents several progressive examples to understand and implement neural networks, starting from a single artificial neuron up to a deep neural network.

Project Contents:
-----------------
1. 1_ArtificialNeuron.ipynb
   - This notebook introduces the concept of an artificial neuron.
   - It covers the basic theory (activation function, weights, bias) and provides a simple implementation.
   - Usage: Run the cells to see how a single neuron processes inputs and produces an output.

2. 2_ArtficialNeuron_CatDog.ipynb
   - This notebook adapts the artificial neuron concept for a "cat vs. dog" classification task.
   - It demonstrates how to adjust weights and bias to differentiate between two classes of images.
   - Usage: Follow the explanations and execute the cells to observe the model's behavior on simple data.

3. 3_2layersNN.ipynb
   - This notebook introduces a neural network with two layers.
   - It implements a more complex architecture to improve classification performance.
   - Usage: Run the notebook to understand the impact of adding a hidden layer on the model's performance.

4. 4_DeepNN.ipynb
   - This notebook presents a deep neural network with multiple layers.
   - It explains modern techniques such as optimization, regularization, and how to handle overfitting.
   - Usage: Follow the notebook to explore current approaches in Deep Learning.

Datasets Folder:
-----------------
The "datasets" folder contains two HDF5 files:
   - trainset.hdf5: The dataset used for training the model.
   - testset.hdf5: The dataset used to evaluate the model's performance.
These files are essential for running the notebooks, as they load the data required for training and testing the models.

utilities.py:
-------------
This file contains common utility functions used by the various notebooks.
It may include data preprocessing, model evaluation, or result visualization functions.
Ensure that any modifications or feature additions are updated in this file to keep the project consistent.

Additional Notes:
-----------------
- Make sure you have installed all necessary dependencies (Python, Jupyter Notebook, and libraries such as NumPy, TensorFlow/PyTorch, h5py, etc.).
- To run the notebooks, launch Jupyter Notebook in the "DeepLearningProject_armftn" directory and open the desired notebook.
- The "__pycache__" folder contains automatically generated compiled Python files and can be ignored.

Enjoy exploring and experimenting with this project!

armftn
