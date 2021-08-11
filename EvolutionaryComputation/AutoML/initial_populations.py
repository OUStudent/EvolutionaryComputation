from EvolutionaryComputation.util import *
from modules import CustomDeepModule, CustomConvolutionModule


class NetworkInitialPopulation:

    def _create_bounds(self):
        if self.convolution_count > 0:
            self.upper_bound_cnn = []
            self.lower_bound_cnn = []
            for j in range(0, self.convolution_count):
                self.upper_bound_cnn.append(1)
                self.lower_bound_cnn.append(0)
                for i in range(0, len(self.convolution_module.layers)):
                    self.upper_bound_cnn.append(1)
                    self.lower_bound_cnn.append(0)
                    if self.convolution_module.layers[i] == "conv":
                        self.upper_bound_cnn.append(1)
                        self.lower_bound_cnn.append(0)
                    elif self.convolution_module.layers[i] == "pooling":
                        self.upper_bound_cnn.append(1)
                        self.lower_bound_cnn.append(0)
                    elif self.convolution_module.layers[i] == 'batchNorm':
                        pass  # no hyper param
                    elif self.convolution_module.layers[i] == "activation":
                        pass
                    elif self.convolution_module.layers[i] == "dropOut":
                        self.upper_bound_cnn.append(self.convolution_module.hyperparam[i][2])
                        self.lower_bound_cnn.append(self.convolution_module.hyperparam[i][1])
            self.total_bounds_cnn = np.asarray(self.upper_bound_cnn) - np.asarray(self.lower_bound_cnn)

        if self.deep_count > 0:
            self.upper_bound_deep = []
            self.lower_bound_deep = []
            for j in range(0, self.deep_count):
                self.upper_bound_deep.append(1)
                self.lower_bound_deep.append(0)
                for i in range(0, len(self.deep_module.layers)):
                    self.upper_bound_deep.append(1)
                    self.lower_bound_deep.append(0)
                    if self.deep_module.layers[i] == "dense":
                        self.upper_bound_deep.append(self.deep_module.hyperparam[i][1])
                        self.lower_bound_deep.append(self.deep_module.hyperparam[i][0])
                    elif self.deep_module.layers[i] == 'batchNorm':
                        pass  # no hyper param
                    elif self.deep_module.layers[i] == "activation":
                        pass
                    elif self.deep_module.layers[i] == "dropOut":
                        self.upper_bound_deep.append(self.deep_module.hyperparam[i][2])
                        self.lower_bound_deep.append(self.deep_module.hyperparam[i][1])
            self.total_bounds_deep = np.asarray(self.upper_bound_deep) - np.asarray(self.lower_bound_deep)

    def _create_population(self):

        if self.convolution_count > 0:
            self.cnn_module_count = int(len(self.lower_bound_cnn) / self.convolution_count)
            self.init_pop_convolution = np.empty(shape=(self.init_size, self.convolution_count,
                                                        self.cnn_module_count))
            for i in range(0, self.init_size):
                for j in range(0, self.convolution_count):
                    for k in range(0, self.cnn_module_count):
                        self.init_pop_convolution[i, j, k] = \
                            np.random.uniform(self.lower_bound_cnn[self.cnn_module_count * j + k],
                                              self.upper_bound_cnn[self.cnn_module_count * j + k], 1)[0]

        if self.deep_count > 0:
            self.deep_module_count = int(len(self.lower_bound_deep) / self.deep_count)
            self.init_pop_deep = np.empty(shape=(self.init_size, self.deep_count,
                                                 self.deep_module_count))
            for i in range(0, self.init_size):
                for j in range(0, self.deep_count):
                    for k in range(0, self.deep_module_count):
                        self.init_pop_deep[i, j, k] = \
                        np.random.uniform(self.lower_bound_deep[self.deep_module_count * j + k],
                                          self.upper_bound_deep[self.deep_module_count * j + k],
                                          1)[0]

    def __init__(self, convolution_module, deep_module, convolution_count, deep_count, init_size, input_shape):
        self.convolution_module = convolution_module
        self.deep_module = deep_module
        self.convolution_count = convolution_count
        self.deep_count = deep_count
        self.init_size = init_size
        self._create_bounds()
        self._create_population()
        self.input_shape = input_shape
        self.fit = None

    def fitness(self, fitness_function):
        self.fit = fitness_function(self)
        return self.fit

    def state_save(self, index):
        state = {"index": index, "obj": self}
        pickle.dump(state, open("state_save{}".format(index), "wb"))

    def create_model(self, index, num_output, output_act):
        model = Sequential()
        if self.convolution_count > 0:
            cnn = self.init_pop_convolution[index].flatten()
            index = 0
            for i in range(0, self.convolution_count):
                if cnn[index] <= self.convolution_module.prob_include:
                    index += 1
                    for layer, hyper in zip(self.convolution_module.layers, self.convolution_module.hyperparam):
                        if layer == "conv":
                            index += 1  # always include conv
                            dist = np.cumsum(hyper[1])
                            for j in range(0, len(hyper[0])):
                                if cnn[index] <= dist[j]:
                                    output_channel = hyper[0][j]
                                    break
                            model.add(
                                Conv2D(output_channel, (3, 3), padding="same",
                                       input_shape=self.input_shape))
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
                                index += 2
                else:
                    index += self.cnn_module_count
            model.add(Flatten())
        if self.deep_count == 0:  # just convo
            model.add(Dense(num_output, activation=output_act))
        else:
            if self.convolution_count == 0:  # just dense
                model.add(Flatten(input_shape=self.input_shape))
            deep = self.init_pop_deep[index].flatten()
            index = 0
            for i in range(0, self.deep_count):
                if deep[index] <= self.deep_module.prob_include:
                    index += 1
                    for layer, hyper in zip(self.deep_module.layers, self.deep_module.hyperparam):
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
                    index += self.deep_module_count
            model.add(Dense(num_output, activation=output_act))
        return model
