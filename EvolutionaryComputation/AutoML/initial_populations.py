from EvolutionaryComputation.util import *
from EvolutionaryComputation.AutoML.modules import CustomDeepModule, CustomConvolutionModule, Saved_Weights

class CustomInitialPopulation:
    """Custom Initial Population by Randomly sampling from domain space for Machine Learning Algorithms

    CustomInitialPopulation is designed for the CustomAutoMLAlgorithm to act as the base initial
    population from which the genetic algorithm can optimize the best individuals. CustomInitialPopulation
    takes in the bounds of the optimization problem and fitness function. Individuals are created
    through random sampling to increase the diversity of the initial population. This diversity is crucial
    for the evolution process as it presents the algorithm with a host of parameters to choose from.

    CustomInitialPopulation implements a "fit" and "plot" method.
     Parameters
    -----------

    fitness_function : function pointer
        A pointer to a function that will evaluate and return the fitness of
        each individual in a population given their parameter values. The function
        should expect two parameters ``generation``, which will be a list of lists,
        where each sub list is an individual; and ``init_pop_print`` which is a
        boolean value defaulted to False which allows the user to print out
        statements during the initial population selection, if chosen (see
        examples for more info on this parameter). Lastly, the function should
        return a numpy array of the fitness values.

    upper_bound : list or numpy 1d array
        A list or numpy 1d array representing the upper bound of the domain for the
        unconstrained problem, where the first index of the list represents the upper
        bound for the first variable. For example, if x1=4, x2=4, x3=8 are the upper
        limits of the variables, then pass in ``[4, 4, 8]`` as the upper bound.

    lower_bound : list or numpy 1d array
        A list or numpy 1d array representing the lower bound of the domain for the
        unconstrained problem, where the first index of the list represents the lower
        bound for the first variable. For example, if x1=0, x2=-4, x3=1 are the lower
        limits of the variables, then pass in ``[0, -4, 1]`` as the lower bound.

    init_size : int
        The number of individuals for the initial population

    Attributes
    -----------
    init_pop : numpy 2D array
        A numpy 2D array of the randomly samples individuals for the initial population

    fitness : numpy 1D array
        A numpy 1D array of the fitness values for each individual
    """

    def __init__(self, upper_bound, lower_bound, init_size):
        self.init_size = init_size
        self.num_variables = len(upper_bound)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.total_bound = np.asarray(upper_bound) - np.asarray(lower_bound)
        self.domain = [upper_bound, lower_bound]
        self.init_pop = None
        self.fit = []

    def fit(self, fitness_function):
        """Obtains the fitness values for the initial population using the `fitness_function`

            Parameters
            -----------

            fitness_function : function pointer
                A pointer to a function that will evaluate and return the fitness of
                each individual in a population given their parameter values. The function
                should expect two parameters ``generation``, which will be a list of lists,
                where each sub list is an individual; and ``init_pop_print`` which is a
                boolean value defaulted to False which allows the user to print out
                statements during the initial population selection, if chosen (see
                examples for more info on this parameter). Lastly, the function should
                return a numpy array of the fitness values.

            """
        self.init_pop = np.empty(shape=(self.init_size, self.num_variables))
        for i in range(0, self.num_variables):
            self.init_pop[:, i] = np.random.uniform(self.lower_bound[i], self.upper_bound[i], self.init_size)

        self.fitness = fitness_function(self.init_pop, init_pop_print=True)

    def plot(self):
        """Plots a box plot of the fitness values after running the initial population
        """
        fig = plt.figure()
        plt.boxplot(self.fitness)
        plt.suptitle("Fitness From Initial Population")
        plt.ylabel("Fitness")
        plt.show()


class NetworkInitialPopulation:

    def load_weights(self, model):
        cnn_mod_count = 0
        deep_mod_count = 0
        for i in range(0, len(model.layers)):
            layer = model.layers[i]
            if isinstance(layer, Conv2D):
                weights = layer.get_weights()
                sph = str(weights[0].shape)
                file = self.save_dir+"/conv2d/" + str(cnn_mod_count) + "_" + sph
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
                file = self.save_dir+"/conv2d/" + str(cnn_mod_count) + "_" + sph
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

    def _create_bounds(self):
        if self.convolution_count > 0:
            self.cnn_module_counts = []
            self.upper_bound_cnn = []
            self.lower_bound_cnn = []
            for j in range(0, self.convolution_count):
                self.upper_bound_cnn.append(1)
                self.lower_bound_cnn.append(0)
                index = 1
                for i in range(0, len(self.convolution_module[j].layers)):
                    self.upper_bound_cnn.append(1)
                    self.lower_bound_cnn.append(0)
                    index += 1
                    if self.convolution_module[j].layers[i] == "conv":
                        self.upper_bound_cnn.append(1)
                        self.lower_bound_cnn.append(0)
                        index += 1
                    elif self.convolution_module[j].layers[i] == "pooling":
                        self.upper_bound_cnn.append(1)
                        self.lower_bound_cnn.append(0)
                        index += 1
                    elif self.convolution_module[j].layers[i] == 'batchNorm':
                        pass  # no hyper param
                    elif self.convolution_module[j].layers[i] == "activation":
                        self.upper_bound_cnn.append(1)
                        self.lower_bound_cnn.append(0)
                        index += 1
                    elif self.convolution_module[j].layers[i] == "dropOut":
                        self.upper_bound_cnn.append(self.convolution_module[j].hyperparam[i][2])
                        self.lower_bound_cnn.append(self.convolution_module[j].hyperparam[i][1])
                        index += 1
                self.cnn_module_counts.append(index)

            self.total_bounds_cnn = np.asarray(self.upper_bound_cnn) - np.asarray(self.lower_bound_cnn)

        if self.deep_count > 0:
            self.deep_module_counts = []
            self.upper_bound_deep = []
            self.lower_bound_deep = []
            for j in range(0, self.deep_count):
                self.upper_bound_deep.append(1)
                self.lower_bound_deep.append(0)
                index = 1
                for i in range(0, len(self.deep_module[j].layers)):
                    self.upper_bound_deep.append(1)
                    self.lower_bound_deep.append(0)
                    index += 1
                    if self.deep_module[j].layers[i] == "dense":
                        self.upper_bound_deep.append(self.deep_module[j].hyperparam[i][1])
                        self.lower_bound_deep.append(self.deep_module[j].hyperparam[i][0])
                        self.upper_bound_deep.append(1)
                        self.lower_bound_deep.append(0)
                        index += 2
                    elif self.deep_module[j].layers[i] == 'batchNorm':
                        pass  # no hyper param
                    elif self.deep_module[j].layers[i] == "activation":
                        pass
                    elif self.deep_module[j].layers[i] == "dropOut":
                        self.upper_bound_deep.append(self.deep_module[j].hyperparam[i][2])
                        self.lower_bound_deep.append(self.deep_module[j].hyperparam[i][1])
                        index += 1
                self.deep_module_counts.append(index)
            self.total_bounds_deep = np.asarray(self.upper_bound_deep) - np.asarray(self.lower_bound_deep)

    def _create_population(self):

        if self.convolution_count > 0:
            self.init_pop_convolution = []
            for i in range(0, self.convolution_count):
                self.init_pop_convolution.append(
                    np.empty(shape=(self.init_size, self.cnn_module_counts[i])))
            self.init_pop_convolution = np.asarray(self.init_pop_convolution)
            for i in range(0, self.init_size):
                for j in range(0, self.convolution_count):
                    for k in range(0, self.cnn_module_counts[j]):
                        self.init_pop_convolution[j][i, k] = \
                            np.random.uniform(self.lower_bound_cnn[int(np.sum(self.cnn_module_counts[:j]) + k)],
                                              self.upper_bound_cnn[int(np.sum(self.cnn_module_counts[:j]) + k)], 1)[0]

        if self.deep_count > 0:
            self.init_pop_deep = []
            for i in range(0, self.deep_count):
                self.init_pop_deep.append(
                    np.empty(shape=(self.init_size, self.deep_module_counts[i])))
            self.init_pop_deep = np.asarray(self.init_pop_deep)
            for i in range(0, self.init_size):
                for j in range(0, self.deep_count):
                    for k in range(0, self.deep_module_counts[j]):
                        self.init_pop_deep[j][i, k] = \
                            np.random.uniform(self.lower_bound_deep[int(np.sum(self.deep_module_counts[:j]) + k)],
                                              self.upper_bound_deep[int(np.sum(self.deep_module_counts[:j]) + k)],
                                              1)[0]

    def __init__(self, convolution_module, deep_module, init_size, input_shape,
                 num_output, output_act, save_dir='saved_weights'):
        self.convolution_module = convolution_module
        self.deep_module = deep_module
        self.convolution_count = len(convolution_module)
        self.deep_count = len(deep_module)
        self.init_size = init_size
        self._create_bounds()
        self._create_population()
        self.input_shape = input_shape
        self.fit = []
        self.save_dir = save_dir
        self.num_output = num_output
        self.output_act = output_act

    def fitness(self, fitness_function, verbose=True, find_max=True, state_save=20, start_index=0):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir+'/conv2d')
            os.makedirs(self.save_dir+'/batch_norm')
            os.makedirs(self.save_dir+'/dense')
        times = []
        bst_ind = 1
        max_models = self.init_size
        for i in range(1+start_index, self.init_size+1):
            if i % state_save == 0:
                if verbose:
                    print("State Save: state_save_init_pop{}".format(i))
                self._state_save(index=i)
            if verbose:
                if i == 1+start_index:
                    msg = "Model 1/{} - Expected Total Time Left: ???".format(self.init_size)
                else:
                    msg = "Model {}/{} - Expected Total Time Left: {} sec".format(i, max_models,
                        np.round(np.mean(times)*(self.init_size-i+1), 4))
                print(msg)
            cnn = self.init_pop_convolution[:, i-1, :].flatten()
            deep = self.init_pop_deep[:, i-1, :].flatten()
            model = self.create_model(cnn=cnn, deep=deep, num_output=self.num_output, output_act=self.output_act)
            self.load_weights(model)
            start = time.time()
            f, v = fitness_function(model)
            finish = time.time()
            self.save_weights(model, val_loss=v)
            self.fit.append(f)
            times.append(finish-start)
            if verbose:
                if find_max:
                    bst = np.round(np.max(self.fit), 4)
                    bst_ind = np.argmax(self.fit)+1
                    top1 = np.round(np.percentile(self.fit, 95), 4)
                    top2 = np.round(np.percentile(self.fit, 90), 4)
                    top3 = np.round(np.percentile(self.fit, 85), 4)
                    top4 = np.round(np.percentile(self.fit, 80), 4)
                else:
                    bst = np.round(np.min(self.fit), 4)
                    top1 = np.round(np.percentile(self.fit, 5), 4)
                    top2 = np.round(np.percentile(self.fit, 10), 4)
                    top3 = np.round(np.percentile(self.fit, 15), 4)
                    top4 = np.round(np.percentile(self.fit, 20), 4)
                    bst_ind = np.argmin(self.fit)+1

                msg = "   Best Model: {} - Best: {}, Top 5%: {}, Top 10%: {}, Top 15%: {}, Top 20%: {}".format(bst_ind, bst, top1,
                                                                                                   top2, top3, top4)
                print(msg)
        self.fit = np.asarray(self.fit)

    def plot(self):
        """Plots a box plot of the fitness values after running the initial population
        """
        fig = plt.figure()
        plt.boxplot(self.fit)
        plt.suptitle("Fitness From Initial Population")
        plt.ylabel("Fitness")
        plt.show()

    def _state_save(self, index):
        state = {"index": index, "obj": self}
        pickle.dump(state, open("state_save_init_pop{}".format(index), "wb"))

    def create_model(self, cnn, deep, num_output, output_act):
        # model = Sequential()
        inputs = tf.keras.Input(shape=self.input_shape)
        first = True
        if self.convolution_count > 0:
            index = 0
            for i in range(0, self.convolution_count):
                if cnn[index] <= self.convolution_module[i].prob_include:
                    index += 1
                    for layer, hyper in zip(self.convolution_module[i].layers, self.convolution_module[i].hyperparam):
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
                    index += self.cnn_module_counts[i]
            x = Flatten()(x)
        if self.deep_count == 0:  # just convo
            outputs = Dense(num_output, activation=output_act)(x)
        else:
            if self.convolution_count == 0:  # just dense
                x = Flatten(input_shape=self.input_shape)(inputs)
            index = 0
            for i in range(0, self.deep_count):
                if deep[index] <= self.deep_module[i].prob_include:
                    index += 1
                    for layer, hyper in zip(self.deep_module[i].layers, self.deep_module[i].hyperparam):
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
                    index += self.deep_module_counts[i]
            outputs = Dense(num_output, activation=output_act)(x)
        return tf.keras.Model(inputs, outputs)
