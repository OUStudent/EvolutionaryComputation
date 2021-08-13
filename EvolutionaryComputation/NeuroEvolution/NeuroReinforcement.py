from NeuroBase import NeuroBase
from EvolutionaryComputation.util import *

class NeuroReinforcer(NeuroBase):

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
        super().plot(self.mean_fit, self.best_fit, None, starting_gen=starting_gen)

    def evolve(self, max_epoch, verbose=True, warm_start=False, algorithm='speciation'):
        if warm_start:
            gen = self.last_gen
            max_epoch += self.prev_epoch
            species = self.species
            species_names = self.species_names
            fit = self._fit
        elif algorithm == 'speciation':
            species, species_names = self._create_species(self.species_layers)
            gen = self._initialize_networks(self.species_layers)
            fit = self._error_function(gen)
        else:
            gen = super()._initialize_networks()
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

            fit_mean = np.mean(fit)
            fit_best = np.max(fit)
            self.mean_fit.append(fit_mean)
            self.best_fit.append(fit_best)

            if verbose:
                msg = "Epoch {}/{}\n" \
                      "  Best Fit: {} Mean Fit: {}".format(k + 1, max_epoch, fit_best, fit_mean)
                print(msg)

            if algorithm == 'lognormal_mutation':
                gen, fit = super()._algorithm_evolutionary_programming(gen, fit, None,
                                                                       reinforcement=self._error_function)
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
                                                                       reinforcement=self._error_function)

        # fit = self._error_function(gen)
        self._fit = fit
        best_index = np.argmax(fit)
        self.best_model = gen[best_index]
        self.last_gen = gen
        self.prev_epoch = max_epoch
        if algorithm == 'speciation':
            self.species = species
            self.species_names = species_names
