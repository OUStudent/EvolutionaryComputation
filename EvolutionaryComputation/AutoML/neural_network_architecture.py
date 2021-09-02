from EvolutionaryComputation.util import *
from EvolutionaryComputation.AutoML.initial_populations import NetworkInitialPopulation
from EvolutionaryComputation.AutoML.modules import Saved_Weights


class NetworkArchitectureEvolution:

    def _crossover_method_1(self, par):
        return np.mean(par, axis=0)

    def _mutation_1_n_z(self, x1, xs, beta):
        return x1 + beta * (xs[0] - xs[1])

    def load_weights(self, model):
        cnn_mod_count = 0
        deep_mod_count = 0
        for i in range(0, len(model.layers)):
            layer = model.layers[i]
            if isinstance(layer, Conv2D):
                weights = layer.get_weights()
                sph = str(weights[0].shape)
                file = self.save_dir + "/conv2d/" + str(cnn_mod_count) + "_" + sph
                if os.path.isfile(file):
                    pre_mod = pickle.load(open(file, "rb"))
                    layer.set_weights(pre_mod.weights)
                cnn_mod_count += 1
            elif isinstance(layer, Dense):
                deep_mod_count += 1
            elif isinstance(layer, BatchNormalization):
                weights = layer.get_weights()
                sph = str(len(weights)) + str(weights[0].shape)
                if deep_mod_count == 0:
                    file = self.save_dir + "/batch_norm/cnn" + str(cnn_mod_count) + "_" + sph
                else:
                    file = self.save_dir + "/batch_norm/deep" + str(deep_mod_count) + "_" + sph
                if os.path.isfile(file):
                    pre_mod = pickle.load(open(file, "rb"))
                    layer.set_weights(pre_mod.weights)

    def save_weights(self, model, val_loss):
        cnn_mod_count = 0
        deep_mod_count = 0
        for layer in model.layers:
            if isinstance(layer, Conv2D):
                weights = layer.get_weights()
                sph = str(weights[0].shape)
                file = self.save_dir + "/conv2d/" + str(cnn_mod_count) + "_" + sph
                if os.path.isfile(file):
                    pre_mod = pickle.load(open(file, "rb"))
                    if pre_mod.val_loss > val_loss:
                        pickle.dump(Saved_Weights(weights=weights, type='conv2d', val_loss=val_loss), open(file, "wb"))
                else:
                    pickle.dump(Saved_Weights(weights=weights, type='conv2d', val_loss=val_loss), open(file, "wb"))
                cnn_mod_count += 1
            elif isinstance(layer, Dense):
                deep_mod_count += 1
            elif isinstance(layer, BatchNormalization):
                weights = layer.get_weights()
                sph = str(len(weights)) + str(weights[0].shape)
                if deep_mod_count == 0:
                    file = self.save_dir + "/batch_norm/cnn" + str(cnn_mod_count) + "_" + sph
                else:
                    file = self.save_dir + "/batch_norm/deep" + str(deep_mod_count) + "_" + sph
                if os.path.isfile(file):
                    pre_mod = pickle.load(open(file, "rb"))
                    if pre_mod.val_loss > val_loss:
                        pickle.dump(Saved_Weights(weights=weights, type='batch_norm', val_loss=val_loss),
                                    open(file, "wb"))
                else:
                    pickle.dump(Saved_Weights(weights=weights, type='batch_norm', val_loss=val_loss), open(file, "wb"))

    def __init__(self, initial_population, gen_size, load_state=None):
        if load_state:
            state = pickle.load(open(load_state, "rb"))
            self.initial_population = state.initial_population
            self.init_size = state.initial_population.init_size
            self.best_fit = state.best_fit
            self.worst_fit = state.worst_fit
            self.mean_fit = state.mean_fit
            self.gen_cnn = state.gen_cnn
            self.gen_deep = state.gen_deep
            self.save_dir = state.save_dir
            self.input_shape = state.input_shape
            self.ensemble = state.ensemble
            self.final_fit = state.final_fit
            self.start_index = state.start_index
            self.num_output = state.num_output
            self.output_act = state.output_act
            self.gen_size = state.initial_population.gen_size
            self.ensemble_fits = state.ensemble_fits
        else:
            self.initial_population = initial_population
            self.init_size = initial_population.init_size
            self.best_fit = []
            self.worst_fit = []
            self.mean_fit = []
            self.gen_cnn = None
            self.gen_deep = None
            self.save_dir = initial_population.save_dir
            self.input_shape = initial_population.input_shape
            self.ensemble = []
            self.final_fit = []
            self.start_index = 0
            self.num_output = initial_population.num_output
            self.output_act = initial_population.output_act
            self.gen_size = gen_size
            self.ensemble_fits = []

    def plot_evolution(self, starting_gen=0):
        """Plots the best and mean fitness values after the evolution process.

        Parameters
        -----------

        starting_gen : int
                      The starting index for plotting.
        """
        x_range = range(starting_gen, len(self.best_fit))
        plt.plot(x_range, self.mean_fit[starting_gen:], label="Mean Fitness")
        plt.plot(x_range, self.best_fit[starting_gen:], label="Best Fitness")
        plt.plot(x_range, self.worst_fit[starting_gen:], label="Worst Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Value")
        plt.suptitle("Best, Mean, and Worst Fitness: ")
        plt.legend()
        plt.show()

    def _create_pop_target(self, target, gen_size, count, counts,
                           mutation_percent, total_bounds, upper_bound, lower_bound):

        init_pop = []
        for i in range(0, count):
            init_pop.append(np.empty(shape=(gen_size, counts[i])))
        for i in range(0, gen_size):
            for j in range(0, count):
                for k in range(0, counts[j]):
                    if i == 0:
                        init_pop[j][0, k] = target[j][0, k]
                    else:
                        idx = int(np.sum(counts[:j]) + k)
                        init_pop[j][i, k] = target[j][0, k] + \
                            np.random.uniform(-mutation_percent * total_bounds[idx],
                                              mutation_percent * total_bounds[idx], 1)[0]
                        if init_pop[j][i, k] > upper_bound[idx]:
                            init_pop[j][i, k] = upper_bound[idx]
                        elif init_pop[j][i, k] < lower_bound[idx]:
                            init_pop[j][i, k] = lower_bound[idx]

        return np.asarray(init_pop)

    def create_model(self, num_output, output_act, cnn=None, deep=None):
        # model = Sequential()
        inputs = tf.keras.Input(shape=self.input_shape)
        first = True
        if self.initial_population.convolution_count > 0:
            index = 0
            for i in range(0, self.initial_population.convolution_count):
                if cnn[index] <= self.initial_population.convolution_module[i].prob_include:
                    index += 1
                    for layer, hyper in zip(self.initial_population.convolution_module[i].layers, self.initial_population.convolution_module[i].hyperparam):
                        if layer == "conv":
                            index += 1  # always include conv
                            dist = np.cumsum(hyper[1])
                            for j in range(0, len(hyper[0])):
                                if cnn[index] <= dist[j]:
                                    output_channel = hyper[0][j]
                                    break

                            if first:
                                x = Conv2D(output_channel, (3, 3), padding='same')(inputs)
                                first = False
                            else:
                                x = Conv2D(output_channel, (3, 3), padding='same')(x)
                            index += 1
                        elif layer == "pooling":
                            if cnn[index] <= hyper[0]:
                                index += 1
                                if hyper[1][0] == 0:
                                    x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
                                elif hyper[1][1] == 0:
                                    x = AveragePooling2D(pool_size=(2, 2), padding="same")(x)
                                elif cnn[index] <= hyper[1][0]:
                                    x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
                                else:
                                    x = AveragePooling2D(pool_size=(2, 2), padding="same")(x)
                                index += 1
                            else:
                                index += 2
                        elif layer == "batchNorm":
                            if cnn[index] <= hyper[0]:
                                index += 1
                                x = BatchNormalization()(x)
                            else:
                                index += 1
                        elif layer == "activation":
                            if cnn[index] <= hyper[0]:
                                index += 1
                                dist = np.cumsum(hyper[2])

                                for j in range(0, len(hyper[1])):
                                    if cnn[index] <= dist[j]:
                                        activation = hyper[1][j]
                                        break
                                if activation == 'leaky_relu':
                                    x = tf.keras.layers.LeakyReLU()(x)
                                else:
                                    x = Activation(activation)(x)
                                index += 1
                            else:
                                index += 2
                        elif layer == 'dropOut':
                            if cnn[index] <= hyper[0]:
                                index += 1
                                dropout = np.round(cnn[index], 2)
                                x = Dropout(dropout)(x)
                                index += 1
                            else:
                                index += 2
                else:
                    index += self.initial_population.cnn_module_counts[i]
            x = Flatten()(x)
        if self.initial_population.deep_count == 0:  # just convo
            outputs = Dense(num_output, activation=output_act)(x)
        else:
            if self.initial_population.convolution_count == 0:  # just dense
                x = Flatten(input_shape=self.input_shape)(inputs)
            index = 0
            for i in range(0, self.initial_population.deep_count):
                if deep[index] <= self.initial_population.deep_module[i].prob_include:
                    index += 1
                    for layer, hyper in zip(self.initial_population.deep_module[i].layers, self.initial_population.deep_module[i].hyperparam):
                        if layer == "dense":
                            index += 1  # always include dense
                            node = np.round(deep[index], 0)
                            index += 1  # node count
                            dist = np.cumsum(hyper[3])
                            for j in range(0, len(hyper[2])):
                                if deep[index] <= dist[j]:
                                    activation = hyper[2][j]
                                    break
                            if activation == 'leaky_relu':
                                x = Dense(node)(x)
                                x = tf.keras.layers.LeakyReLU()(x)
                            else:
                                x = Dense(node, activation=activation)(x)
                            index += 1  # activation
                        elif layer == "batchNorm":
                            if deep[index] <= hyper[0]:
                                index += 1
                                x = BatchNormalization()(x)
                            else:
                                index += 1
                        elif layer == 'dropOut':
                            if deep[index] <= hyper[0]:
                                index += 1
                                dropout = np.round(deep[index], 2)
                                x = Dropout(dropout)(x)
                                index += 1
                            else:
                                index += 2
                else:
                    index += self.initial_population.deep_module_counts[i]
            outputs = Dense(num_output, activation=output_act)(x)
        return tf.keras.Model(inputs, outputs)

    def _state_save(self, index):
        state = {"index": index, "obj": self}
        pickle.dump(state, open("state_save_evolution{}".format(index), "wb"))

    def evolve(self, max_iter, fitness_function,
               verbose=True, find_max=False, state_save=5, warm_start=False):

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir + '/conv2d')
            os.makedirs(self.save_dir + '/batch_norm')
            os.makedirs(self.save_dir + '/dense')

        if not warm_start:
            if find_max:
                temp = np.argsort(-self.initial_population.fit)[0:self.gen_size]
            else:
                temp = np.argsort(self.initial_population.fit)[0:self.gen_size]

            if self.initial_population.convolution_count > 0:
                self.gen_cnn = np.asarray(self.initial_population.init_pop_convolution)[:,temp,:]

            if self.initial_population.deep_count > 0:
                self.gen_deep = np.asarray(self.initial_population.init_pop_deep)[:, temp, :]

            fit = self.initial_population.fit[temp]
        else:
            fit = self.final_fit
            self.gen_size = len(self.gen_deep[0, :, 0])

        fit_mean = np.mean(fit)
        if find_max:
            fit_best = np.max(fit)
            fit_worst = np.min(fit)
        else:
            fit_best = np.min(fit)
            fit_worst = np.max(fit)
        self.mean_fit.append(fit_mean)
        self.best_fit.append(fit_best)
        self.worst_fit.append(fit_worst)
        if verbose:
            msg = "Initial Generation : " \
                  "\n  Best Fit: {}, Mean Fit: {}, Worst Fit: {}".format(fit_best, fit_mean, fit_worst)
            print(msg)

        beta = 0.2
        times = []
        max_models = (self.start_index + max_iter)*self.gen_size
        first = True
        for k in range(self.start_index, self.start_index + max_iter):
            if k % state_save == 0 and k != self.start_index:
                if verbose:
                    msg = "State Save: state_save_evolution{}".format(k-1)
                    print(msg)

                self._state_save(index=k+1)

            if verbose:
                if first:
                    msg = "Generation {} - Model {}/{} - Expected Total Time Left: ???".format(k, 0, self.gen_size)
                    first = False
                else:
                    msg = "Generation {} - Model {}/{} - Expected Total Time Left: {} sec".format(k, 0, self.gen_size,
                                                                                            np.round(np.mean(times) * max_models,
                                                                                          4))
                sys.stdout.write("\r" + msg)
            self.start_index += 1

            ch_cnn = []
            ch_deep = []
            for i in range(0, self.gen_size):
                child_cnn2 = None
                child_deep2 = None
                ind = np.random.choice(range(0, self.gen_size), 3, replace=False)
                if self.initial_population.convolution_count > 0:
                    par_cnn = []
                    targ_cnn = []
                    diff_cnn_1 = []
                    diff_cnn_2 = []
                    for j in range(0, self.initial_population.convolution_count):
                        par_cnn.append(self.gen_cnn[j][i])
                        targ_cnn.append(self.gen_cnn[j][ind[2]])
                        diff_cnn_1.append(self.gen_cnn[j][ind[0]])
                        diff_cnn_2.append(self.gen_cnn[j][ind[1]])
                    par_cnn = np.asarray(par_cnn)
                    targ_cnn = np.asarray(targ_cnn)
                    diff_cnn_1 = np.asarray(diff_cnn_1)
                    diff_cnn_2 = np.asarray(diff_cnn_2)
                    unit_cnn = self._mutation_1_n_z(targ_cnn, [diff_cnn_1, diff_cnn_2], beta)
                    for l in range(0, len(unit_cnn)):
                        for j in range(0, len(unit_cnn[l])):
                            idx = int(np.sum(self.initial_population.cnn_module_counts[:l]) + j)
                            if unit_cnn[l, j] > self.initial_population.upper_bound_cnn[idx]:
                                unit_cnn[l, j] = self.initial_population.upper_bound_cnn[idx]
                            elif unit_cnn[l, j] < self.initial_population.lower_bound_cnn[idx]:
                                unit_cnn[l, j] = self.initial_population.lower_bound_cnn[idx]
                    child_cnn = self._crossover_method_1([par_cnn, unit_cnn])
                    ch_cnn.append(child_cnn)
                    child_cnn2 = child_cnn.flatten()

                if self.initial_population.deep_count > 0:
                    par_deep = []
                    targ_deep = []
                    diff_deep_1 = []
                    diff_deep_2 = []
                    for j in range(0, self.initial_population.deep_count):
                        par_deep.append(self.gen_deep[j][i])
                        targ_deep.append(self.gen_deep[j][ind[2]])
                        diff_deep_1.append(self.gen_deep[j][ind[0]])
                        diff_deep_2.append(self.gen_deep[j][ind[1]])
                    par_deep = np.asarray(par_deep)
                    targ_deep = np.asarray(targ_deep)
                    diff_deep_1 = np.asarray(diff_deep_1)
                    diff_deep_2 = np.asarray(diff_deep_2)
                    unit_deep = self._mutation_1_n_z(targ_deep, [diff_deep_1, diff_deep_2], beta)
                    for l in range(0, len(unit_deep)):
                        for j in range(0, len(unit_deep[l])):
                            idx = int(np.sum(self.initial_population.deep_module_counts[:l]) + j)
                            if unit_deep[l, j] > self.initial_population.upper_bound_deep[idx]:
                                unit_deep[l, j] = self.initial_population.upper_bound_deep[idx]
                            elif unit_deep[l, j] < self.initial_population.lower_bound_deep[idx]:
                                unit_deep[l, j] = self.initial_population.lower_bound_deep[idx]
                    child_deep = self._crossover_method_1([par_deep, unit_deep])
                    ch_deep.append(child_deep)
                    child_deep2 = child_deep.flatten()

                model = self.create_model(num_output=self.num_output, output_act=self.output_act,
                                          deep=child_deep2, cnn=child_cnn2)

                self.load_weights(model)
                strt = time.time()
                f, v = fitness_function(model)
                finish = time.time()
                self.save_weights(model, val_loss=v)
                times.append(finish-strt)
                if verbose:
                    msg = "Generation {} - Model {}/{} - Expected Total Time Left: {} sec".format(k, i+1, self.gen_size,
                                                                                            np.round(np.mean(times) * max_models,
                                                                                          4))
                    if i == self.gen_size-1:
                        msg = msg + '\n'
                    sys.stdout.write("\r" + msg)
                max_models -= 1
                if find_max:
                    if f > fit[np.argmin(fit)]:
                        fit[np.argmin(fit)] = f
                        if self.initial_population.convolution_count > 0:
                            self.gen_cnn[:, np.argmin(fit), :] = child_cnn
                        if self.initial_population.deep_count > 0:
                            self.gen_deep[:,np.argmin(fit),:] = child_deep
                else:
                    if f < fit[np.argmax(fit)]:
                        fit[np.argmax(fit)] = f
                        if self.initial_population.convolution_count > 0:
                            self.gen_cnn[:, np.argmax(fit), :] = child_cnn
                        if self.initial_population.deep_count > 0:
                            self.gen_deep[:, np.argmax(fit), :] = child_deep

            fit_mean = np.mean(fit)
            if find_max:
                fit_best = np.max(fit)
                fit_worst = np.min(fit)
            else:
                fit_best = np.min(fit)
                fit_worst = np.max(fit)
            self.mean_fit.append(fit_mean)
            self.best_fit.append(fit_best)
            self.worst_fit.append(fit_worst)
            self.final_fit = fit
            if verbose:
                msg = "  Best Fit: {}, Mean Fit: {}, Worst Fit: {}".format(fit_best, fit_mean, fit_worst)
                print(msg)

    def create_ensemble(self, fitness_function, find_max=False, verbose=True):
        times = []
        self.ensemble = []
        self.ensemble_fits = []
        for i in range(0, self.gen_size):
            if verbose:
                if i == 0:
                    msg = "Model 1/{} - Expected Total Time Left: ???".format(self.gen_size)
                else:
                    msg = "Model {}/{} - Expected Total Time Left: {} sec".format(i+1, self.gen_size,
                                                                            np.round(np.mean(times) * (
                                                                                        self.gen_size - i), 4))
                print(msg)
            cnn = self.gen_cnn[:, i, :].flatten()
            deep = self.gen_deep[:, i, :].flatten()
            model = self.create_model(num_output=self.num_output, output_act=self.output_act,
                                      deep=deep, cnn=cnn)
            self.load_weights(model)
            start = time.time()
            f, v = fitness_function(model)
            finish = time.time()
            times.append(finish - start)
            self.save_weights(model, val_loss=v)
            self.ensemble.append(model)
            self.ensemble_fits.append(f)
            if find_max:
                bst = np.max(self.ensemble_fits)
                bst_ind = np.argmax(self.ensemble_fits) + 1
                wst = np.min(self.ensemble_fits)
            else:
                bst = np.min(self.ensemble_fits)
                bst_ind = np.argmin(self.ensemble_fits) + 1
                wst = np.max(self.ensemble_fits)

            if verbose:
                msg = "Best Model: {}, Best: {}, Mean: {}, Worst: {}".format(bst_ind, bst, np.mean(self.ensemble_fits), wst)
                print(msg)


    def plot_ensemble(self):
        fig = plt.figure()
        plt.boxplot(self.ensemble_fits)
        plt.suptitle("Fitness From Ensemble")
        plt.ylabel("Fitness")
        plt.show()

    def ensemble_predict(self, x_data, method='sum'):
        preds = np.asarray([model.predict(x_data) for model in self.ensemble])
        if method == 'sum':
            vals = np.sum(preds, axis=0)
        elif method == 'mean':
            vals = np.mean(preds, axis=0)
        elif method == 'median':
            vals = np.mean(preds, axis=0)
        else:
            vals = [0]
        return vals

    def ensemble_predict_classes(self, x_data, method='sum'):
        preds = np.asarray([model.predict(x_data) for model in self.ensemble])
        if method == 'sum':
            vals = np.sum(preds, axis=0)
        elif method == 'mean':
            vals = np.mean(preds, axis=0)
        elif method == 'median':
            vals = np.mean(preds, axis=0)
        else:
            vals = [0]
        return np.argmax(vals, axis=1)
