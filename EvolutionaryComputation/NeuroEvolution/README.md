# NeuroEvolution

__________
This submodule contains the specialized Genetic Algorithms for evolving the weights of a feed forward neural network. 
As of current, there exist four types of feed forward neural networks: classification `NeuroClassifier`, regression `NeuroRegressor`, 
auto-encoders `NeuroAutoEncoder`, and reinforcement learning.

# Network Architecture

The neural network architecture supports multiple layers, nodes, and activation functions. It must be stated that only 
the weights of the layers are evolved, not the layer or node counts. 

For classification, regression, and auto-encoders there exists three algorithms: `greedy`, `generic`, and `self-adaptive`. The `greedy` algorithm creates 8 offspring while
the `generic` and `self-adaptive` algorithm only creates 4.



### Warning - Non Reinforcement Algorithms

The algorithms for evolving the weights of a neural network are designed for small network sizes. Genetic algorithms work 
well at optimizing a small amount of parameters; however, standard networks can reach sizes up to one million parameters. 
It is suggested that the maximum number of trainable parameters allowed should be around 10,000 to 15,000 before performance 
becomes unusable. 

This module should not be used on real-world datasets with many data observations and input parameters. It is suggest that 
this submodule is purely academic and experimental. 

# Reinforcement Learning Algorithms

For reinforcement learning, the algorithms work by evolving the weights and activation functions for a feed forward 
neural network. Unlike `NeuroClassifier`, `NeuroRegressor`, and `NeuroAutoEncoder`, the reinforcement learning algorithms
have great success at real-world applications. There exist two main classes: `NeuroReinforcer` and `NeuroReinforcerImages`.
`NeuroReinforcer` is designed for numerical input while `NeuroReinforcerImages` is designed to handle images as input. 
For reinforcement learning, there exist two algorithms: `self-adaptive` and `speciation`. The `self-adaptive` algorithm 
evolves the weights of the network with static activation functions while `speciation` allows for the algorithm to evolve 
the activations functions for each layer.

`NeuroReinforcerImages` is not built off keras, unlike the automated machine learning algorithm: `NetworkArchitectureEvolution`.
Instead, `NeuroReinforcerImages` is built off an advanced CPU convolution library called `CNNumpy`,
developed by `3outellie`: [link to repository](https://github.com/3outeille/CNNumpy).