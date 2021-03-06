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
        self.species_present = []
        super().__init__(layer_nodes=layer_nodes, num_input=num_input, num_output=num_output,
                         activation_function=activation_function, output_activation=output_activation,
                         error_function=fitness_function, population_size=population_size, reinforcement=True)

    def plot(self, starting_gen=0, plot_species=False):
        """Plots the best and mean fitness values after the evolution process.

        Parameters
        -----------

        starting_gen : int
                      The starting index for plotting.
        """

        if plot_species:
            x = range(starting_gen, len(self.mean_fit))
            species_present = list(self.species_layers[0].keys())
            for key in species_present:
                counts = []
                for layers in self.species_layers:
                    if key in layers:
                        counts.append(layers[key])
                    else:
                        counts.append(0)
                plt.plot(x, counts, label=key)

            plt.xlabel("Epochs/Generations")
            plt.ylabel("Species Size")
            plt.suptitle("Species Sizes After Evolution")
            plt.legend()
            plt.show()
        else:
            super().plot(self.mean_fit, self.best_fit, None, starting_gen=starting_gen)

    def evolve(self, max_epoch, verbose=True, warm_start=False, algorithm='speciation', just_layers=False,
               prob_chnge_species=None):
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
            species_layers = self.species_layers
        elif algorithm == 'speciation':
            species, species_names = self._create_species(self.species_layers)
            gen = self._initialize_networks(self.species_layers, just_layers=just_layers)
            fit = self._error_function(gen)
            species_layers = []
        else:
            gen = super()._initialize_networks(just_layers=just_layers)
            fit = self._error_function(gen)
            species_layers = []

        if verbose:
            num_param = self.num_input * self.layer_nodes[0] + self.layer_nodes[0]
            for i in range(1, len(self.layer_nodes)):
                num_param += self.layer_nodes[i - 1] * self.layer_nodes[i] + self.layer_nodes[i]
            num_param += self.num_output * self.layer_nodes[len(self.layer_nodes) - 1] + 1
            if not warm_start:
                msg = "Number of Trainable Parameters Per Network: {}".format(num_param)
                print(msg)

        for k in range(self.prev_epoch, max_epoch):

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

                species_present = []
                for i in range(0, len(gen)):
                    name = ",".join(gen[i].activation_function_name)
                    species_present.append(name)

                species_present, spec_count = np.unique(species_present, return_counts=True)

                # if reinforce -> check np.argmax in msg
                msg = '  Number of Species Present: {}\n' \
                      '    Best Species by Top Fit: {}\n'.format(len(species_present),
                                                                 gen[np.argmax(fit)].activation_function_name)
                present = {}
                for i in range(0, len(species_present)):
                    msg += "    Species: [{}] Count: {}\n".format(species_present[i], spec_count[i])
                    present[species_present[i]] = spec_count[i]
                species_layers.append(present)
                if verbose:
                    print(msg[:-1])  # skip last '\n'

                if prob_chnge_species is not None:
                    for i in range(0, len(gen)):
                        if species_layers[k][(",".join(gen[i].activation_function_name))] == 1:
                            continue
                        r = np.random.uniform(0, 1, 1)[0]
                        if r <= prob_chnge_species:
                            if just_layers:
                                spec = list(species_layers[0].keys())
                                v = list(range(0, len(species_layers[0])))
                            else:
                                spec = species_names
                                v = list(range(0, len(species_names)))
                            cur_idx = spec.index(",".join(gen[i].activation_function_name))

                            v.pop(cur_idx)
                            r = np.random.choice(v)
                            index = 0
                            for j in range(0, len(spec)):
                                if r == j:
                                    break
                                index += 1

                            new_spec = spec[index].split(',')
                            if verbose:
                                msg = "     Change from Species {} to Species {}".format(gen[i].activation_function_name, new_spec)
                                print(msg)
                            gen[i].activation_function_name = new_spec
                            gen[i].activation_function = []
                            for fun in new_spec:
                                if fun == 'relu':
                                    gen[i].activation_function.append(relu)
                                elif fun == 'tanh':
                                    gen[i].activation_function.append(tanh)
                                elif fun == 'sigmoid':
                                    gen[i].activation_function.append(sigmoid)
                                elif fun == 'unit':
                                    gen[i].activation_function.append(unit)
                                elif fun == 'purlin':
                                    gen[i].activation_function.append(purlin)
                                elif fun == 'softmax':
                                    gen[i].activation_function.append(softmax)
                                elif fun == 'gaussian':
                                    gen[i].activation_function.append(gaussian)
                                elif fun == 'elu':
                                    gen[i].activation_function.append(elu)
                                elif fun == 'leaky_relu':
                                    gen[i].activation_function.append(leaky_relu)
                                elif fun == 'selu':
                                    gen[i].activation_function.append(selu)

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
            self.species_layers = species_layers
