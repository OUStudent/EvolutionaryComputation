from EvolutionaryComputation.util import *

class NeuroBase:

    def __init__(self, layer_nodes, num_input, num_output, error_function, activation_function='relu',
                 output_activation='purlin', population_size=20, reinforcement=False):
        self.best_fit = []
        self.mean_fit = []
        self.val_fit = []
        self.layer_nodes = layer_nodes
        self.num_input = num_input
        self.num_output = num_output
        self._error_function = error_function
        self.activation_function_name = activation_function
        self.output_activation_name = output_activation
        self._max_gen_size = population_size
        self.last_gen = None
        self.best_model = None
        self.species = None
        self.species_names = None
        self.prev_epoch = 0
        self.reinforcement = reinforcement

    def _calculate_relative_fit(self, fit, tourn_fit, find_max):
        s = 0
        r = np.random.uniform(0, 1, len(tourn_fit))
        for i in range(0, len(tourn_fit)):
            if fit == tourn_fit[i]:  # if fitness is equal, 50/50 chance of being better
                if r[i] >= 0.5:
                    s += 1
            else:
                if find_max:
                    if fit > tourn_fit[i]:
                        s += 1
                else:
                    if fit < tourn_fit[i]:
                        s += 1
        return s

    def _scale_fitness1(self, x):
        return 1 / (1 + x)

    def _scale_fitness2(self, x):
        return x + np.abs(np.min(x))

    def __roulette_wheel_selection(self, cumulative_sum, n):
        ind = []
        r = np.random.uniform(0, 1, n)
        for i in range(0, n):
            index = 0
            while cumulative_sum[index] < r[i]:
                index += 1
            ind.append(index)
        return ind

    def __fitness_function(self, gen, data):
        output = []
        input_data = data[0]
        expected_output = data[1]
        for i in range(0, len(gen)):
            t = self._error_function(expected_output, gen[i].predict(input_data))
            output.append(t)
        return np.asarray(output)

    def _initialize_networks(self, speciation=None, just_layers=False):
        init_gen = []
        for i in range(0, self._max_gen_size):
            if speciation:
                num_layers = len(self.layer_nodes)
                if just_layers:
                    activations = [np.random.choice(speciation, 1)[0]]*num_layers
                else:
                    activations = np.random.choice(speciation, num_layers).tolist()
                obj = self.EvolvableNetwork(layer_nodes=self.layer_nodes, num_input=self.num_input,
                                            num_output=self.num_output,
                                            activation_function=activations,
                                            output_activation=self.output_activation_name, initialize=True)
            else:
                obj = self.EvolvableNetwork(layer_nodes=self.layer_nodes, num_input=self.num_input,
                                            num_output=self.num_output, activation_function=self.activation_function_name,
                                            output_activation=self.output_activation_name, initialize=True)
            init_gen.append(obj)
        return init_gen

    def __crossover_1(self, p1, p2, const_cross):
        # initialize new network with empty layer weights and biases
        child = self.EvolvableNetwork(layer_nodes=p1.layer_nodes, num_input=p1.num_input, num_output=p1.num_output,
                                      activation_function=self.activation_function_name,
                                      output_activation=self.output_activation_name, initialize=False)
        # fill child weight and bias matrices from the parents
        for i in range(0, p1.layer_count + 1):
            child.layer_weights.append((1 - const_cross) * p1.layer_weights[i] + const_cross * p2.layer_weights[i])
            child.layer_biases.append((1 - const_cross) * p1.layer_biases[i] + const_cross * p2.layer_biases[i])
            child.sigmas.append((1 - const_cross) * p1.sigmas[i] + const_cross * p2.sigmas[i])
        return child

    def __crossover_2(self, p1, p2):
        child = self.EvolvableNetwork(layer_nodes=p1.layer_nodes, num_input=p1.num_input, num_output=p1.num_output,
                                      activation_function=self.activation_function_name,
                                      output_activation=self.output_activation_name, initialize=False)
        random_nums = np.random.randint(low=0, high=2, size=p1.layer_count + 1)
        for i in range(0, p1.layer_count + 1):
            if random_nums[i] == 0:
                child.layer_weights.append(np.copy(p1.layer_weights[i]))
                child.layer_biases.append(np.copy(p1.layer_biases[i]))
                child.sigmas.append(np.copy(p1.sigmas[i]))
            else:
                child.layer_weights.append(np.copy(p2.layer_weights[i]))
                child.layer_biases.append(np.copy(p2.layer_biases[i]))
                child.sigmas.append(np.copy(p2.sigmas[i]))
        return child

    def __mutation_lognormal(self, par):
        child = copy.deepcopy(par)
        for i in range(0, par.layer_count + 1):
            n, c = child.layer_weights[i].shape
            child.sigmas[i] += np.random.uniform(-0.01 * child.sigmas[i], 0.01 * child.sigmas[i], 1)[0]
            #tau = 1 / (np.sqrt(2 * np.sqrt(n*c)))
            #tau_prime = 1 / (np.sqrt(2 * n*c))
            #r = np.random.normal(0, 1, 2)
            #child.sigmas[i] = child.sigmas[i] * np.exp(tau * r[0] + tau_prime * r[1])
            child.layer_weights[i] += np.random.uniform(-child.sigmas[i], child.sigmas[i], n * c).reshape(n, c)
            if i == par.layer_count:
                child.layer_biases[i] += np.random.uniform(-child.sigmas[i], child.sigmas[i], 1)
            else:
                child.layer_biases[i] += np.random.uniform(-child.sigmas[i], child.sigmas[i], c)
        return child

    def __mutation(self, par):
        child = copy.deepcopy(par)

        for i in range(0, par.layer_count + 1):
            n, c = child.layer_weights[i].shape
            child.layer_weights[i] += np.random.uniform(-child.sigmas[i], child.sigmas[i], n * c).reshape(n, c)
            if i == par.layer_count:
                child.layer_biases[i] += np.random.uniform(-child.sigmas[i], child.sigmas[i], 1)
            else:
                child.layer_biases[i] += np.random.uniform(-child.sigmas[i], child.sigmas[i], c)
        return child

    def __reproduce_all(self, p1, p2):
        # if cross_method == 'average':
        c_cross = np.random.normal(0.5, 0.15, 2)
        ch1 = self.__crossover_1(p1, p2, c_cross[0])
        ch2 = self.__crossover_1(p1, p2, c_cross[1])
        # elif cross_method == 'intuitive':
        ch3 = self.__crossover_2(p1, p2)
        ch4 = self.__crossover_2(p1, p2)
        ch5 = self.__mutation(ch1)
        ch6 = self.__mutation(ch2)
        ch7 = self.__mutation(ch3)
        ch8 = self.__mutation(ch4)
        all = [ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8]
        return all

    def __reproduce(self, p1, p2, train_data_batch, cross_method, reinforcement=None):
        if cross_method == 'average':
            c_cross = np.random.normal(0.5, 0.15, 2)
            ch1 = self.__crossover_1(p1, p2, c_cross[0])
            ch2 = self.__crossover_1(p1, p2, c_cross[1])
        elif cross_method == 'intuitive':
            ch1 = self.__crossover_2(p1, p2)
            ch2 = self.__crossover_2(p1, p2)
        ch3 = self.__mutation(ch1)
        ch4 = self.__mutation(ch2)
        all = [ch1, ch2, ch3, ch4]
        if reinforcement is None:
            fit = self.__fitness_function(all, train_data_batch)
            return all[np.argmin(fit)], np.min(fit)
        else:
            fit = reinforcement(all)
            return all[np.argmax(fit)], np.max(fit)

    def __random_selection(self, n, count):
        return np.random.choice(range(0, n), count)

    def _algorithm_generic(self, gen, fit, batch_data, reinforcement=None):
        if reinforcement:
            scaled_fit = self._scale_fitness2(fit)
        else:
            scaled_fit = self._scale_fitness1(fit)
        n = len(gen)
        best_index = np.maximum(2, int(n*0.05))
        best_indices = np.argsort(-scaled_fit)[0:best_index]
        og_fit = np.copy(fit)
        fit_sum = np.sum(scaled_fit)
        fit = scaled_fit / fit_sum
        cumulative_sum = np.cumsum(fit)
        selected = self.__roulette_wheel_selection(cumulative_sum, n)  # self.__random_selection(n, n)
        mates = self.__roulette_wheel_selection(cumulative_sum, n)  # self.__random_selection(n, n)
        children = []
        children_fit = []
        for i in range(0, n):
            child, child_fit = self.__reproduce(gen[selected[i]], gen[mates[i]], batch_data, 'average', reinforcement)
            children.append(child)
            children_fit.append(child_fit)
        children_fit = np.asarray(children_fit)
        if reinforcement:
            worst_fits = np.argsort(children_fit)[0:best_index]
        else:
            worst_fits = np.argsort(-children_fit)[0:best_index]
        worst_fits = worst_fits.tolist()
        for best in reversed(best_indices):
            index = 0
            for worst in worst_fits:
                if reinforcement:
                    if children_fit[worst] < og_fit[best]:
                        break
                else:
                    if children_fit[worst] > og_fit[best]:
                        break
                index += 1
            if index != len(worst_fits):
                children[worst_fits[index]] = gen[best]
                del worst_fits[index]
        return children

    def _algorithm_greedy(self, gen, fit, batch_data, reinforcement=None):
        n = len(gen)
        selected = self.__random_selection(n, n)
        mates = self.__random_selection(n, n)
        children = []
        for i in range(0, n):
            childs = self.__reproduce_all(gen[selected[i]], gen[mates[i]])
            for child in childs:
                children.append(child)
        if reinforcement is None:
            child_fit = self.__fitness_function(children, batch_data)
        else:
            child_fit = reinforcement(children)
        all_fit = np.concatenate([fit, child_fit])
        all_models = gen + children
        if reinforcement:
            best_fits = np.argsort(-all_fit)[0:n]
        else:
            best_fits = np.argsort(all_fit)[0:n]
        return [all_models[i] for i in best_fits]

    def _algorithm_evolutionary_programming(self, gen, fit, batch_data, reinforcement=None, num_offspring=4):

        offspring_gen = []
        n = len(gen)
        for i in range(0, n):
            for j in range(0, num_offspring):
                offspring_gen.append(self.__mutation_lognormal(gen[i]))

        if isinstance(gen, list):
            total = gen + offspring_gen
        else:
            total = gen.tolist() + offspring_gen

        if reinforcement is None:
            offspring_fit = self.__fitness_function(offspring_gen, batch_data)
        else:
            offspring_fit = reinforcement(total)

        ind = np.asarray(range(0, len(total)))
        if reinforcement:
            temp = offspring_fit
        else:
            temp = np.concatenate([fit, offspring_fit])
        if reinforcement:
            ind = ind[np.argsort(-temp)]
        else:
            ind = ind[np.argsort(temp)]

        if reinforcement:
            return np.asarray(total, dtype=object)[ind][0:n], temp[ind][0:n]
        else:
            return [total[i] for i in ind[0:n]]

    def _create_species_names(self, layer_count, activations, s, species):
        if layer_count > 0:
            for i in range(0, len(activations)):
                k = s + "," + activations[i]
                self._create_species_names(layer_count-1, activations, k, species)
        else:
            species.append(s[1:])

    def _create_species(self, layers):
        species = {}
        id = 0
        species_names = []
        self._create_species_names(len(self.layer_nodes), layers, "", species_names)
        for s in species_names:
            species[s] = id
            id += 1
        return species, species_names

    def evolve(self, max_epoch, batch_size, train_data, val_data, early_stopping=True,
               verbose=True, warm_start=False, algorithm='generic', patience=10):
        if warm_start:
            gen = self.last_gen
            species = self.species
            species_names = self.species_names
        elif algorithm == 'speciation':
            layers = ['relu', 'tanh', 'sigmoid', 'gaussian', 'leaky_relu']
            species, species_names = self._create_species(layers)
            gen = self._initialize_networks(layers)
        else:
            gen = self._initialize_networks()
        prev_val = 100000
        val_index = 0
        ind = np.arange(0, len(train_data[1]), 1)
        batches_train = []
        for i in range(0, len(ind), batch_size):
            batches_train.append(ind[i:i + batch_size])
        ind = np.arange(0, len(val_data[1]), 1)
        batches_val = []
        for i in range(0, len(ind), batch_size):
            batches_val.append(ind[i:i + batch_size])

        if warm_start and max_epoch != -1:
            max_epoch = max_epoch + self.prev_epoch

        if max_epoch == -1:
            max_epoch = 100000
            run = True
        else:
            run = False

        if verbose:
            num_param = self.num_input*self.layer_nodes[0]+self.layer_nodes[0]
            for i in range(1, len(self.layer_nodes)):
                num_param += self.layer_nodes[i-1]*self.layer_nodes[i]+self.layer_nodes[i]
            num_param += self.num_output*self.layer_nodes[len(self.layer_nodes)-1]+1
            if not warm_start:
                msg = "Number of Trainable Parameters Per Network: {}".format(num_param)
                print(msg)
        num_batches = len(batches_train)
        for k in range(self.prev_epoch, max_epoch):

            if run and k == max_epoch - 1:
                max_epoch = max_epoch * 2

            local_mean_fit = []
            local_best_fit = []
            local_val_mean = []
            if verbose:
                if run:
                    msg = "Epoch {}/???".format(k + 1)
                else:
                    msg = "Epoch {}/{}".format(k + 1, max_epoch)
                print(msg)
            batch_index = 0
            for g in range(0, num_batches):
                batch_train = batches_train[g]
                batch_val = batches_val[g % len(batches_val)]
                fit = self.__fitness_function(gen, [train_data[0][batch_train], train_data[1][batch_train]])
                fit_mean = np.mean(fit)
                fit_best = np.min(fit)

                if g == 0 and algorithm == 'speciation':
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
                                                                         gen[np.argmin(fit)].activation_function_name)
                        keys = list(species.keys())
                        for index in species_present:
                            msg += "    Species: [{}] Count: {}\n".format(keys[index], spec_count[index])
                        print(msg[:-1])  # skip last '\n'

                local_mean_fit.append(fit_mean)
                local_best_fit.append(fit_best)
                val_fit = self.__fitness_function(gen, [val_data[0][batch_val], val_data[1][batch_val]])
                val_mean = np.mean(val_fit)
                local_val_mean.append(val_mean)

                if verbose:
                    msg = " Batch {}/{}" \
                          "   Results for Batch: Best Loss: {} Mean Loss: {} Val Loss: {}".format(batch_index + 1, num_batches,
                                                                               np.round(fit_best, 7), np.round(fit_mean, 7),
                                                                                                  np.round(val_mean, 7))
                    if batch_index == num_batches - 1:
                        msg = msg + '\n'
                    sys.stdout.write("\r" + msg)
                batch_index += 1

                if algorithm == 'self-adaptive' or algorithm == 'speciation':
                    gen = self._algorithm_evolutionary_programming(gen, fit, [train_data[0][batch_train],
                                                                               train_data[1][batch_train]])
                elif algorithm == 'generic':
                    gen = self._algorithm_generic(gen, fit,
                                                   [train_data[0][batch_train], train_data[1][batch_train]])

                elif algorithm == 'greedy':
                    gen = self._algorithm_greedy(gen, fit,
                                                   [train_data[0][batch_train], train_data[1][batch_train]])

            self.mean_fit.append(np.mean(local_mean_fit))
            self.best_fit.append(np.min(local_best_fit))
            self.val_fit.append(np.mean(local_val_mean))
            if self.val_fit[k] > prev_val:
                val_index += 1
            else:
                val_index = 0
                prev_val = self.val_fit[k]
            if val_index == patience:
                if early_stopping:
                    print("Ending Evolution: Over Fitting")
                    break


        val_fit = self.__fitness_function(gen, val_data)
        best_index = np.argmin(val_fit)
        if algorithm == 'speciation':
            self.species = species
            self.species_names = species_names
        self.best_model = gen[best_index]
        self.last_gen = gen
        self.prev_epoch = max_epoch

    def plot(self, mean_fit, best_fit, val_fit=None, starting_gen=0):
        x = range(starting_gen, len(mean_fit))
        plt.plot(x, mean_fit[starting_gen:], label="Mean Loss")
        plt.plot(x, best_fit[starting_gen:], label="Best Loss")
        if val_fit:
            plt.plot(x, val_fit[starting_gen:], label="Validation Loss")
        plt.xlabel("Epochs/Generations")
        plt.ylabel("Loss")
        plt.suptitle("Loss Scores After Evolution")
        plt.legend()
        plt.show()

    class EvolvableNetwork:

        def __init__(self, layer_nodes, num_input, num_output, activation_function='relu',
                     output_activation='purlin', initialize=True):
            self.layer_count = len(layer_nodes)
            self.layer_nodes = layer_nodes
            self.num_input = num_input
            self.num_output = num_output
            self.activation_function = []
            if activation_function == 'relu':
                self.activation_function_name = 'relu'
                for i in range(0, self.layer_count + 1):
                    self.activation_function.append(relu)
            elif activation_function == 'tanh':
                self.activation_function_name = 'tanh'
                for i in range(0, self.layer_count + 1):
                    self.activation_function.append(tanh)
            elif activation_function == 'sigmoid':
                self.activation_function_name = 'sigmoid'
                for i in range(0, self.layer_count + 1):
                    self.activation_function.append(sigmoid)
            elif activation_function == 'unit':
                self.activation_function_name = 'unit'
                for i in range(0, self.layer_count + 1):
                    self.activation_function.append(unit)
            elif activation_function == 'purlin':
                self.activation_function_name = 'purlin'
                for i in range(0, self.layer_count + 1):
                    self.activation_function.append(purlin)
            elif activation_function == 'softmax':
                self.activation_function_name = 'softmax'
                for i in range(0, self.layer_count + 1):
                    self.activation_function.append(softmax)
            elif activation_function == 'gaussian':
                self.activation_function_name = 'gaussian'
                for i in range(0, self.layer_count + 1):
                    self.activation_function.append(gaussian)
            elif activation_function == 'elu':
                self.activation_function_name = 'elu'
                for i in range(0, self.layer_count + 1):
                    self.activation_function.append(elu)
            elif activation_function == 'leaky_relu':
                self.activation_function_name = 'leaky_relu'
                for i in range(0, self.layer_count + 1):
                    self.activation_function.append(leaky_relu)
            elif activation_function == 'selu':
                self.activation_function_name = 'selu'
                for i in range(0, self.layer_count + 1):
                    self.activation_function.append(selu)
            else:
                self.activation_function_name = []
                for fun in activation_function:
                    if fun == 'relu':
                        self.activation_function_name.append("relu")
                        self.activation_function.append(relu)
                    elif fun == 'tanh':
                        self.activation_function_name.append("tanh")
                        self.activation_function.append(tanh)
                    elif fun == 'sigmoid':
                        self.activation_function_name.append("sigmoid")
                        self.activation_function.append(sigmoid)
                    elif fun == 'unit':
                        self.activation_function_name.append("unit")
                        self.activation_function.append(unit)
                    elif fun == 'purlin':
                        self.activation_function_name.append("purlin")
                        self.activation_function.append(purlin)
                    elif fun == 'softmax':
                        self.activation_function_name.append("softmax")
                        self.activation_function.append(softmax)
                    elif fun == 'gaussian':
                        self.activation_function_name.append("gaussian")
                        self.activation_function.append(gaussian)
                    elif fun == 'elu':
                        self.activation_function_name.append("elu")
                        self.activation_function.append(elu)
                    elif fun == 'leaky_relu':
                        self.activation_function_name.append("leaky_relu")
                        self.activation_function.append(leaky_relu)
                    elif fun == 'selu':
                        self.activation_function_name.append('selu')
                        self.activation_function.append(selu)

            self.output_activation_name = output_activation
            if output_activation == 'sigmoid':
                self.output_activation = sigmoid
            elif output_activation == 'purlin':
                self.output_activation = purlin
            elif output_activation == 'softmax':
                self.output_activation = softmax
            elif output_activation == 'tanh':
                self.output_activation = tanh

            self.layer_weights = []
            self.layer_biases = []
            self.sigmas = []
            if not initialize:  # I will discuss the purpose of this later
                return

            r = np.random.uniform(0.01, 0.2, self.layer_count + 1)

            # create the NxM weight and bias matrices for input Layer
            limit_w = np.sqrt(6 / (num_output + layer_nodes[0]))
            self.sigmas.append(limit_w * r[0])
            self.layer_weights.append(
                np.random.uniform(-limit_w, limit_w, num_input * layer_nodes[0]).reshape(num_input, layer_nodes[0]))
            self.layer_biases.append(np.random.uniform(-limit_w, limit_w, layer_nodes[0]))

            # create the weight matrices for Hidden Layers
            for i in range(1, self.layer_count):
                limit_w = np.sqrt(6 / (layer_nodes[i - 1] + layer_nodes[i]))
                self.sigmas.append(limit_w * r[i])
                self.layer_weights.append(
                    np.random.uniform(-limit_w, limit_w, layer_nodes[i - 1] * layer_nodes[i]).reshape(
                        layer_nodes[i - 1], layer_nodes[i]))
                self.layer_biases.append(
                    np.random.uniform(-limit_w, limit_w, layer_nodes[i]).reshape(1, layer_nodes[i]))

            # Create the weight and bias matrices for output Layer
            limit_w = np.sqrt(6 / (layer_nodes[self.layer_count - 1] + num_output))
            self.sigmas.append(limit_w * r[self.layer_count])
            self.layer_weights.append(
                np.random.uniform(-limit_w, limit_w, layer_nodes[self.layer_count - 1] * num_output).reshape(
                    layer_nodes[self.layer_count - 1],
                    num_output))
            self.layer_biases.append(np.random.uniform(-limit_w, limit_w, num_output).reshape(1, num_output))

        def predict(self, x, encode=False, decode=False):  # same as forward pass, performs matrix multiplication of the weights

            if encode:
                output = self.activation_function[0](np.dot(x, self.layer_weights[0]) + self.layer_biases[0])
                if self.layer_count == 1:
                    return output
                for i in range(1, int((self.layer_count+1)/2)):
                    if i == self.layer_count:  # last layer so don't use activation function
                        output = (np.dot(output, self.layer_weights[i]) + self.layer_biases[i])
                    else:
                        output = self.activation_function[i](
                            np.dot(output, self.layer_weights[i]) + self.layer_biases[i])
                return output
            elif decode:
                output = x
                if self.layer_count == 1:
                    output = np.dot(output, self.layer_weights[1])+self.layer_biases[1]
                    return output
                for i in range(int((self.layer_count+1)/2), self.layer_count+1):
                    if i == self.layer_count:  # last layer so don't use activation function
                        output = (np.dot(output, self.layer_weights[i]) + self.layer_biases[i])
                    else:
                        output = self.activation_function[i](
                            np.dot(output, self.layer_weights[i]) + self.layer_biases[i])
                if self.num_output == 1:  # if there is only one output variable then reshape
                    return self.output_activation(output).reshape(len(x), )
                return self.output_activation(output)
            else:
                output = self.activation_function[0](np.dot(x, self.layer_weights[0]) + self.layer_biases[0])
                for i in range(1, self.layer_count + 1):
                    if i == self.layer_count:  # last layer so don't use activation function
                        output = (np.dot(output, self.layer_weights[i]) + self.layer_biases[i])
                    else:
                        output = self.activation_function[i](
                            np.dot(output, self.layer_weights[i]) + self.layer_biases[i])
                if self.num_output == 1:  # if there is only one output variable then reshape
                    return self.output_activation(output).reshape(len(x), )
                return self.output_activation(output)



