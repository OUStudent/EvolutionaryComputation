from NeuroBase import NeuroBase
from EvolutionaryComputation.util import *

class NeuroReinforcer(NeuroBase):
    """Feed Forward Neural Network for Reinforcement Learning whose weights and activation functions are evolved.

            NeuroReinforcer evolves the weights of a static feed forward neural network
            given the hidden layer/node count along with activation functions. It also
            implements an "evolve", "predict", and "plot" method.

            Unlike NeuroReinforcerImages, NeuroReinforcer is designed for non-image like
            numerical input. If input is images, please use NeuroReinforcerImages instead.

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

            activation_function : string or list, default = None
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
    def __init__(self, layer_nodes, num_input, num_output, fitness_function, activation_function=None,
                 population_size=100, output_activation='softmax'):
        if activation_function is None:
            activation_function = ['relu', 'tanh', 'sigmoid', 'gaussian', 'leaky_relu', 'selu']
        self.species_layers = activation_function
        self._fit = None
        super().__init__(layer_nodes=layer_nodes, num_input=num_input, num_output=num_output,
                         activation_function=activation_function, output_activation=output_activation,
                         error_function=fitness_function, population_size=population_size, reinforcement=True)

    def plot(self, starting_gen=0):
        """Plots the best and mean fitness values after the evolution process.

        Parameters
        -----------

        starting_gen : int
                      The starting index for plotting.
        """
        super().plot(self.mean_fit, self.best_fit, None, starting_gen=starting_gen)

    def evolve(self, max_epoch, verbose=True, warm_start=False, algorithm='speciation', just_layers=False):
        """Perform evolution with the given set of parameters

        Parameters
        -----------

        max_epoch : int
                 The maximum number of epochs to run the evolution process

        just_layers : bool, default = False
                If True and algorithm == 'speciation', then each model will
                be created whose activation functions per layer will be equal,
                but randomly sampled from the possible activation functions
                passed in through init. If False and algorithm == 'speciation',
                then each model will be created whose activation functions will
                be different by being randomly sampled from the possible activation
                functions passed in through init. See example notebook for more info.

        verbose : bool, default = True
                If True, print out information during the evolution process

        warm_start : bool
                    If True, the algorithm will use the last generation
                    from the previous generation instead of creating a
                    new initial population

        algorithm : string, default = 'speciation'
               A string to denote the algorithm to be used for evolution.
               Two algorithms are currently available: 'speciation' and
               'self-adaptive'. Please see the example notebooks for a
               run down.

        """
        if warm_start:
            gen = self.last_gen
            max_epoch += self.prev_epoch
            species = self.species
            species_names = self.species_names
            fit = self._fit
        elif algorithm == 'speciation':
            species, species_names = self._create_species(self.species_layers)
            gen = self._initialize_networks(self.species_layers, just_layers=just_layers)
            fit = self._error_function(gen)
        else:
            gen = super()._initialize_networks(just_layers=just_layers)
            fit = self._error_function(gen)

        if verbose:
            num_param = self.num_input * self.layer_nodes[0] + self.layer_nodes[0]
            for i in range(1, len(self.layer_nodes)):
                num_param += self.layer_nodes[i - 1] * self.layer_nodes[i] + self.layer_nodes[i]
            num_param += self.num_output * self.layer_nodes[len(self.layer_nodes) - 1] + 1
            if not warm_start:
                msg = "Number of Trainable Parameters Per Network: {}".format(num_param)
                print(msg)

        for k in range(self.prev_epoch, max_epoch):

            print("Training Model: CPU: {}, RAM: {}, Memory: {}".format(psutil.cpu_percent(),
                                                                        psutil.virtual_memory().percent,
                                                                        psutil.Process(
                                                                            os.getpid()).memory_info().rss / 1024 ** 2))

            fit_mean = np.mean(fit)
            fit_best = np.max(fit)
            self.mean_fit.append(fit_mean)
            self.best_fit.append(fit_best)

            if verbose:
                msg = "Epoch {}/{}\n" \
                      "  Best Fit: {} Mean Fit: {}".format(k + 1, max_epoch, fit_best, fit_mean)
                print(msg)

            if algorithm == 'self-adaptive':
                gen, fit = super()._algorithm_evolutionary_programming(gen, fit, None,
                                                                       reinforcement=self._error_function,
                                                                       num_offspring=2)
            elif algorithm == 'generic':
                gen = super()._algorithm_generic(gen, fit, None, reinforcement=self._error_function)
            elif algorithm == 'greedy':
                gen = super()._algorithm_greedy(gen, fit, None, reinforcement=self._error_function)
            elif algorithm == 'speciation':
                if verbose:
                    species_present = []
                    for i in range(0, len(gen)):
                        name = ""
                        for act in gen[i].activation_function_name:
                            name = name + ',' + act
                        species_present.append(name[1:])
                    spec_count = []
                    species_present = np.asarray(species_present)
                    for spec in species_names:
                        spec_count.append(np.count_nonzero(np.where(species_present == spec)))
                    species_present = np.where(np.asarray(spec_count) > 0)[0]
                    # if reinforce -> check np.argmax in msg
                    msg = '  Number of Species Present: {}\n' \
                          '    Best Species by Top Fit: {}\n'.format(len(species_present),
                                                                     gen[np.argmax(fit)].activation_function_name)
                    keys = list(species.keys())
                    for index in species_present:
                        msg += "    Species: [{}] Count: {}\n".format(keys[index], spec_count[index])
                    print(msg[:-1])  # skip last '\n'

                gen, fit = super()._algorithm_evolutionary_programming(gen, fit, None,
                                                                       reinforcement=self._error_function,
                                                                       num_offspring=2)

        # fit = self._error_function(gen)
        self._fit = fit
        best_index = np.argmax(fit)
        self.best_model = gen[best_index]
        self.last_gen = gen
        self.prev_epoch = max_epoch
        if algorithm == 'speciation':
            self.species = species
            self.species_names = species_names
