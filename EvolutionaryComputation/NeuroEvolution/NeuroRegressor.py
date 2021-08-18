from NeuroBase import NeuroBase
from EvolutionaryComputation.util import *

class NeuroRegressor(NeuroBase):
    """Feed Forward Neural Network for Regression whose weights are evolved.

        NeuroRegressor evolves the weights of a static feed forward neural network
        given the hidden layer/node count along with activation functions. It also
        implements an "evolve", "predict", and "plot" method.

        Parameters
        ------------

        layer_nodes : list of integers
                   A list containing the number of nodes for each hidden layer. For
                   example, if `layer_nodes` = `[10, 25, 10]` then there will be
                   three hidden layers with node counts 10, 15, 10.

        num_input : int
                  The number of input variables.

        num_output : int
                    The number of output variables.

        activation_function : string or list, default = 'relu'
                            If string, all the hidden layers will have the same activation
                            function. If list, the hidden layers will get the corresponding
                            activation function. For example, if `activation_function`
                            equals `['relu', 'tanh', 'sigmoid']` then the first hidden layer will
                            get 'relu', while the second 'tanh', and third 'sigmoid'. The currently
                            available activation functions are : 'relu', 'tanh', 'sigmoid', 'selu',
                            'elu', 'unit', 'softmax', 'gaussian', 'leaky_relu', and 'purlin' (identity).

        population_size : int, default = 100
                      The population size of the generation.

        Attributes
        -----------

        best_model : EvolvableNetwork Object
            An EvolvableNetwork Object representing the best individual from the last generation
            of evolution.

        """
    def __init__(self, layer_nodes, num_input, num_output, activation_function="relu",
                 population_size=100):
        super().__init__(layer_nodes=layer_nodes, num_input=num_input, num_output=num_output,
                         activation_function=activation_function, output_activation='purlin',
                         error_function=mse_error, population_size=population_size)

    def evolve(self, max_epoch, batch_size, train_data, val_data, early_stopping=True,
               verbose=True, warm_start=False, algorithm='generic', patience=10):
        """Perform evolution with the given set of parameters

        Parameters
        -----------

        max_epoch : int
                 The maximum number of epochs to run the evolution process

        batch_size : int
                 The size of each batch before evolving the networks

        train_data : list of data
                 A list of the training data, where the first index is the input
                 and the second index is the expected output. Example:
                 `train_data` = [x_train, y_train].

        val_data : list of data
                 A list of the validation data, where the first index is the input
                 and the second index is the expected output. Example:
                 `val_data` = [x_val, y_val].

        early_stopping : bool, default = True
                        If True, the algorithm will perform early stopping when the
                        error on the validation data does not decrease after `patience`
                        amount of epochs.

        patience : int, default = 10
                 The number of epochs from which early stopping will be performed if
                 the error does not decrease on the validation error. Value of `patience`
                 does not matter if `early_stopping` is False.

        verbose : bool, default = True
                If True, print out information during the evolution process

        warm_start : bool
                    If True, the algorithm will use the last generation
                    from the previous generation instead of creating a
                    new initial population

        algorithm : string, default = 'greedy'
               A string to denote the algorithm to be used for evolution.
               Three algorithms are currently available: 'greedy', 'generic',
               and 'self-adaptive'. Please see the example notebooks for a
               run down.

        """
        super().evolve(max_epoch=max_epoch, batch_size=batch_size, train_data=train_data, val_data=val_data,
                       early_stopping=early_stopping, verbose=verbose, warm_start=warm_start,
                       algorithm=algorithm, patience=patience)

    def predict(self, data):
        """Predicts the outcome of the data using the best model

        Parameters
        -----------

        data : numpy 2D array
             The data to be used as input into the best model, number of columns
             must equal the number of input variables and the number of rows
             refers to the number of data observations.
        """
        return self.best_model.predict(data)

    def plot(self, starting_gen=0):
        """Plots the best and mean fitness values after the evolution process.

        Parameters
        -----------

        starting_gen : int
                      The starting index for plotting.
        """
        super().plot(self.mean_fit, self.best_fit, self.val_fit, starting_gen=starting_gen)
