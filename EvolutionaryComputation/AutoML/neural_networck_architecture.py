from EvolutionaryComputation.util import *
from initial_populations import NetworkInitialPopulation

class NetworkArchitectureEvolution:

    def _crossover_method_1(self, par):
        return np.mean(par, axis=0)

    def _mutation_1_n_z(self, x1, xs, beta):
        return x1 + beta * (xs[0] - xs[1])

    def __init__(self, initial_population, fitness_function):
        self.initial_population = initial_population
        self.init_size = initial_population.init_size
        self.best_fit = []
        self.worst_fit = []
        self.mean_fit = []
        self.fitness_function = fitness_function
        self.gen_cnn = None
        self.gen_deep = None

    def _create_pop_target(self, target, gen_size, mutation_percent, module_size, total_bounds):
        init_pop = np.empty(
            shape=(gen_size, len(target), module_size))
        for i in range(0, gen_size):
            for j in range(0, len(target)):
                for k in range(0, module_size):
                    init_pop[i, j, k] = target[j, k] + \
                                                         np.random.uniform(
                                                             -mutation_percent * total_bounds[
                                                                 module_size * j + k],
                                                             mutation_percent * total_bounds[
                                                                 module_size * j + k],
                                                             1)[0]

        return init_pop

    def create_model(self, num_output, output_act, cnn=None, deep=None):
        model = Sequential()

        if self.initial_population.convolution_count > 0:
            index = 0
            for i in range(0, self.initial_population.convolution_count):
                if cnn[index] <= self.initial_population.convolution_module.prob_include:
                    index += 1
                    for layer, hyper in zip(self.initial_population.convolution_module.layers,
                                            self.initial_population.convolution_module.hyperparam):
                        if layer == "conv":
                            index += 1  # always include conv
                            dist = np.cumsum(hyper[1])
                            for j in range(0, len(hyper[0])):
                                if cnn[index] <= dist[j]:
                                    output_channel = hyper[0][j]
                                    break
                            model.add(
                                Conv2D(output_channel, (3, 3), padding="same",
                                       input_shape=self.initial_population.input_shape))
                            index += 1
                        elif layer == "pooling":
                            if cnn[index] <= hyper[0]:
                                index += 1
                                if hyper[1][0] == 0:
                                    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
                                elif hyper[1][1] == 0:
                                    model.add(AveragePooling2D(pool_size=(2, 2), padding="same"))
                                elif cnn[index] <= hyper[1][0]:
                                    model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
                                else:
                                    model.add(AveragePooling2D(pool_size=(2, 2), padding="same"))
                                index += 1
                            else:
                                index += 1
                        elif layer == "batchNorm":
                            if cnn[index] <= hyper[0]:
                                index += 1
                                model.add(BatchNormalization())
                            else:
                                index += 1
                        elif layer == "activation":
                            if cnn[index] <= hyper[0]:
                                index += 1
                                model.add(Activation(hyper[1]))
                            else:
                                index += 1
                        elif layer == 'dropOut':
                            if cnn[index] <= hyper[0]:
                                index += 1
                                dropout = np.round(cnn[index], 2)
                                model.add(Dropout(dropout))
                                index += 1
                            else:
                                index += 1
                else:
                    index += self.initial_population.cnn_module_count
            model.add(Flatten())
        if self.initial_population.deep_count == 0:  # just convo
            model.add(Dense(num_output, activation=output_act))
        else:
            if self.initial_population.convolution_count == 0:  # just dense
                model.add(Flatten(input_shape=self.initial_population.input_shape))
            index = 0
            for i in range(0, self.initial_population.deep_count):
                if deep[index] <= self.initial_population.deep_module.prob_include:
                    index += 1
                    for layer, hyper in zip(self.initial_population.deep_module.layers,
                                            self.initial_population.deep_module.hyperparam):
                        if layer == "dense":
                            index += 1  # always include dense
                            node = np.round(deep[index], 0)
                            model.add(Dense(node, activation=hyper[2]))
                            index += 1
                        elif layer == "batchNorm":
                            if deep[index] <= hyper[0]:
                                index += 1
                                model.add(BatchNormalization())
                            else:
                                index += 1
                        elif layer == 'dropOut':
                            if deep[index] <= hyper[0]:
                                index += 1
                                dropout = np.round(deep[index], 2)
                                model.add(Dropout(dropout))
                                index += 1
                            else:
                                index += 2
                else:
                    index += self.initial_population.deep_module_count
            model.add(Dense(num_output, activation=output_act))
        return model

    def evolve(self, max_models, num_outputs, output_act, percent=0.8, gen_size=None, max_iter=None, num_best=1,
               mutation_percent=0.1, verbose=True, find_max=False, state_save=5):
        if gen_size is None:
            gen_size = np.maximum(5, int((max_models - percent * max_models) * 0.035))
        if max_iter is None:
            max_iter = np.maximum(5, int((max_models - percent * max_models) / gen_size))

        if find_max:
            temp = np.argsort(-self.initial_population.fit)[0:num_best]
        else:
            temp = np.argsort(self.initial_population.fit)[0:num_best]

        if self.initial_population.convolution_count > 0:
            best_cnn = self.initial_population.init_pop_convolution[temp]
            if num_best == 1:
                target_cnn = best_cnn[0]
            else:
                target_cnn = self._crossover_method_1(best_cnn)
            self.gen_cnn = self._create_pop_target(target_cnn, gen_size, mutation_percent, self.initial_population.cnn_module_count,
                                              self.initial_population.total_bounds_cnn)
            self.gen_cnn[0] = target_cnn
        if self.initial_population.deep_count > 0:
            best_deep = self.initial_population.init_pop_deep[temp]
            if num_best == 1:
                target_deep = best_deep[0]
            else:
                target_deep = self._crossover_method_1(best_deep)
            self.gen_deep = self._create_pop_target(target_deep, gen_size, mutation_percent, self.initial_population.deep_module_count,
                                               self.initial_population.total_bounds_deep)
            self.gen_deep[0] = target_deep

        fit = []
        for i in range(0, gen_size):
            if self.initial_population.deep_count > 0 and self.initial_population.convolution_count > 0:
                model = self.create_model(num_output=num_outputs, output_act=output_act, cnn=self.gen_cnn[i].flatten(),
                                      deep=self.gen_deep[i].flatten())
            elif self.initial_population.convolution_count > 0:
                model = self.create_model(num_output=num_outputs, output_act=output_act, cnn=self.gen_cnn[i].flatten(),
                                          deep=None)
            else:
                model = self.create_model(num_output=num_outputs, output_act=output_act, cnn=None,
                                          deep=self.gen_deep[i].flatten())

            fit.append(self.fitness_function(model))
        beta = 0.2
        for k in range(0, max_iter):
            fit_mean = np.mean(fit)
            if find_max:
                fit_best = np.max(fit)
                fit_worst = np.min(fit)
            else:
                fit_best = np.min(fit)
                fit_worst = np.max(fit)
            self.mean_fit.append(fit_mean)
            self.best_fit.append(fit_best)
            if verbose:
                msg = "GENERATION {}:\n" \
                      "  Best Fit: {}, Mean Fit: {}, Worst Fit: {}".format(k, fit_best, fit_mean, fit_worst)
                print(msg)

            '''if (k + 1) % state_save == 0:
                print("State Save")
                pickle.dump(self, open("state_save{}".format(k), "wb"))'''
            ch_cnn = []
            ch_deep = []
            ch_fit = []
            for i in range(0, gen_size):

                ind = np.random.choice(range(0, gen_size), 3, replace=False)
                if self.initial_population.convolution_count > 0:
                    par_cnn = self.gen_cnn[i]
                    targ_cnn = self.gen_cnn[ind[2]]
                    unit_cnn = self._mutation_1_n_z(targ_cnn, self.gen_cnn[ind[0:2]], beta)
                    for l in range(0, len(unit_cnn)):
                        for j in range(0, len(unit_cnn[0])):
                            if unit_cnn[l, j] > self.initial_population.upper_bound_cnn[len(unit_cnn)*l+j]:
                                unit_cnn[l, j] = self.initial_population.upper_bound_cnn[len(unit_cnn)*l+j]
                            elif unit_cnn[l, j] < self.initial_population.lower_bound_cnn[len(unit_cnn)*l+j]:
                                unit_cnn[l, j] = self.initial_population.lower_bound_cnn[len(unit_cnn)*l+j]
                    child_cnn = self._crossover_method_1([par_cnn, unit_cnn])
                    ch_cnn.append(child_cnn)
                if self.initial_population.deep_count > 0:
                    par_deep = self.gen_deep[i]
                    targ_deep = self.gen_deep[ind[2]]
                    unit_deep = self._mutation_1_n_z(targ_deep, self.gen_deep[ind[0:2]], beta)
                    child_deep = self._crossover_method_1([par_deep, unit_deep])
                    for l in range(0, len(child_deep)):
                        for j in range(0, len(child_deep[0])):
                            if child_deep[l, j] > self.initial_population.upper_bound_deep[len(child_deep[0])*l+j]:
                                child_deep[l, j] = self.initial_population.upper_bound_deep[len(child_deep[0])*l+j]
                            elif child_deep[l, j] < self.initial_population.lower_bound_deep[len(child_deep[0])*l+j]:
                                child_deep[l, j] = self.initial_population.lower_bound_deep[len(child_deep[0])*l+j]

                    ch_deep.append(child_deep)
                if self.initial_population.deep_count > 0 and self.initial_population.convolution_count > 0:
                    model = self.create_model(num_output=num_outputs, output_act=output_act, cnn=child_cnn.flatten(),
                                              deep=child_deep.flatten())
                elif self.initial_population.convolution_count > 0:
                    model = self.create_model(num_output=num_outputs, output_act=output_act, cnn=child_cnn.flatten(),
                                              deep=None)
                else:
                    model = self.create_model(num_output=num_outputs, output_act=output_act, cnn=None,
                                              deep=child_deep.flatten())
                f = self.fitness_function(model)


                if find_max:
                    if f > fit[np.argmin(fit)]:
                        fit[np.argmin(fit)] = f
                        if self.initial_population.convolution_count > 0:
                            self.gen_cnn[np.argmin(fit)] = child_cnn
                        if self.initial_population.deep_count > 0:
                            self.gen_deep[np.argmin(fit)] = child_deep
                else:
                    if f < fit[np.argmax(fit)]:
                        fit[np.argmax(fit)] = f
                        if self.initial_population.convolution_count > 0:
                            self.gen_cnn[np.argmax(fit)] = child_cnn
                        if self.initial_population.deep_count > 0:
                            self.gen_deep[np.argmax(fit)] = child_deep


                #ch_fit.append(f)

            '''all_fit = np.concatenate((fit, ch_fit))
            if find_max:
                bst = np.argsort(-all_fit)[0:gen_size]
            else:
                bst = np.argsort(all_fit)[0:gen_size]
            if self.initial_population.convolution_count > 0:
                self.gen_cnn = np.concatenate((self.gen_cnn, ch_cnn))[bst]
            if self.initial_population.deep_count > 0:
                self.gen_deep = np.concatenate((self.gen_deep, ch_deep))[bst]
            fit = all_fit[bst]'''
