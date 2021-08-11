from EvolutionaryComputation.util import *
from NeuroBase import NeuroBase

def get_indices(X_shape, HF, WF, stride, pad):
    """
        Returns index matrices in order to transform our input image into a matrix.
        Parameters:
        -X_shape: Input image shape.
        -HF: filter height.
        -WF: filter width.
        -stride: stride value.
        -pad: padding value.
        Returns:
        -i: matrix of index i.
        -j: matrix of index j.
        -d: matrix of index d.
            (Use to mark delimitation for each channel
            during multi-dimensional arrays indexing).
    """
    # get input size
    m, n_C, n_H, n_W = X_shape

    # get output size
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1

    # ----Compute matrix of index i----

    # Level 1 vector.
    level1 = np.repeat(np.arange(HF), WF)
    # Duplicate for the other channels.
    level1 = np.tile(level1, n_C)
    # Create a vector with an increase by 1 at each level.
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # Create matrix of index i at every levels for each channel.
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # ----Compute matrix of index j----

    # Slide 1 vector.
    slide1 = np.tile(np.arange(WF), HF)
    # Duplicate for the other channels.
    slide1 = np.tile(slide1, n_C)
    # Create a vector with an increase by 1 at each slide.
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # Create matrix of index j at every slides for each channel.
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # ----Compute matrix of index d----

    # This is to mark delimitation for each channel
    # during multi-dimensional arrays indexing.
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d


def im2col(X, HF, WF, stride, pad):
    """
        Transforms our input image into a matrix.
        Parameters:
        - X: input image.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.
        Returns:
        -cols: output matrix.
    """
    # Padding
    X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    # Multi-dimensional arrays indexing.
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols


def col2im(dX_col, X_shape, HF, WF, stride, pad):
    """
        Transform our matrix back to the input image.
        Parameters:
        - dX_col: matrix with error.
        - X_shape: input image shape.
        - HF: filter height.
        - WF: filter width.
        - stride: stride value.
        - pad: padding value.
        Returns:
        -x_padded: input image with error.
    """
    # Get input size
    N, D, H, W = X_shape
    # Add padding if needed.
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))

    # Index matrices, necessary to transform our input image into a matrix.
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    # Retrieve batch dimension by spliting dX_col N times: (X, Y) => (N, X, Y)
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    # Reshape our matrix back to image.
    # slice(None) is used to produce the [::] effect which means "for every elements".
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    # Remove padding from new image if needed.
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]

class Conv():

    def __init__(self, nb_filters, filter_size, nb_channels, stride=1, padding=0):
        self.n_F = nb_filters
        self.f = filter_size
        self.n_C = nb_channels
        self.s = stride
        self.p = padding

        # Xavier-Glorot initialization - used for sigmoid, tanh.
        self.W = {'val': np.random.randn(self.n_F, self.n_C, self.f, self.f) * np.sqrt(1. / (self.f)),
                  'grad': np.zeros((self.n_F, self.n_C, self.f, self.f))}
        self.b = {'val': np.random.randn(self.n_F) * np.sqrt(1. / self.n_F), 'grad': np.zeros((self.n_F))}

        self.cache = None

    def forward(self, X):
        """
            Performs a forward convolution.

            Parameters:
            - X : Last conv layer of shape (m, n_C_prev, n_H_prev, n_W_prev).
            Returns:
            - out: previous layer convolved.
        """
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_C = self.n_F
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        X_col = im2col(X, self.f, self.f, self.s, self.p)
        w_col = self.W['val'].reshape((self.n_F, -1))
        b_col = self.b['val'].reshape(-1, 1)
        # Perform matrix multiplication.
        out = w_col @ X_col + b_col
        # Reshape back matrix to image.
        out = np.array(np.hsplit(out, m)).reshape((m, n_C, n_H, n_W))
        self.cache = X, X_col, w_col
        return out

class MaxPool():

    def __init__(self, filter_size, stride=1, padding=0):
        self.f = filter_size
        self.s = stride
        self.p = padding
        self.cache = None

    def forward(self, X):
        """
            Apply average pooling.
            Parameters:
            - X: Output of activation function.

            Returns:
            - A_pool: X after average pooling layer.
        """
        self.cache = X

        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        n_C = n_C_prev
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        X_col = im2col(X, self.f, self.f, self.s, self.p)
        X_col = X_col.reshape(n_C, X_col.shape[0] // n_C, -1)
        A_pool = np.max(X_col, axis=1)
        # Reshape A_pool properly.
        A_pool = np.array(np.hsplit(A_pool, m))
        A_pool = A_pool.reshape(m, n_C, n_H, n_W)

        return A_pool

class AvgPool():

    def __init__(self, filter_size, stride=1, padding=0):
        self.f = filter_size
        self.s = stride
        self.p = padding
        self.cache = None

    def forward(self, X):
        """
            Apply average pooling.
            Parameters:
            - X: Output of activation function.

            Returns:
            - A_pool: X after average pooling layer.
        """
        self.cache = X

        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        n_C = n_C_prev
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1

        X_col = im2col(X, self.f, self.f, self.s, self.p)
        X_col = X_col.reshape(n_C, X_col.shape[0] // n_C, -1)
        A_pool = np.mean(X_col, axis=1)
        # Reshape A_pool properly.
        A_pool = np.array(np.hsplit(A_pool, m))
        A_pool = A_pool.reshape(m, n_C, n_H, n_W)

        return A_pool


import copy

class NeuroReincorferImages(NeuroBase):

    class EvolvableCNN(NeuroBase):

        def __init__(self, cnn_layers, picture_dim, deep_layers, deep_activation, deep_out, num_output):
            self.layers = []
            self.cnn_activations = []
            self.sigmas = []
            index = 0
            num_train = 0
            for layer in cnn_layers:
                if 'Conv' in layer:
                    num_filter, filter_size = layer[layer.find("(") + 1:layer.find(")")].replace(" ", "").split(",")
                    num_filter = int(num_filter)
                    filter_size = int(filter_size)
                    val = np.sqrt(1.0/filter_size)

                    self.sigmas.append(np.random.uniform(0.01*val, 0.2*val, 1)[0])
                    if index == 0:
                        nb_channel = picture_dim[2]
                    else:
                        nb_channel = prev_filter_size
                    num_train += filter_size * filter_size * num_filter * nb_channel + filter_size
                    self.layers.append(Conv(nb_filters=num_filter, filter_size=filter_size,
                                            nb_channels=nb_channel))
                    prev_filter_size = num_filter
                elif 'AvgPool' in layer:
                    filter_size, stride = layer[layer.find("(") + 1:layer.find(")")].replace(" ", "").split(",")
                    self.layers.append(AvgPool(filter_size=int(filter_size), stride=int(stride)))
                elif 'MaxPool' in layer:
                    filter_size, stride = layer[layer.find("(") + 1:layer.find(")")].replace(" ", "").split(",")
                    self.layers.append(MaxPool(filter_size=int(filter_size), stride=int(stride)))
                else:
                    if layer == 'sigmoid':
                        self.layers.append(sigmoid)
                    elif layer == 'tanh':
                        self.layers.append(tanh)
                    elif layer == 'gaussian':
                        self.layers.append(gaussian)
                    elif layer == 'relu':
                        self.layers.append(relu)
                    elif layer == 'selu':
                        self.layers.append(selu)
                    elif layer == 'leaky_relu':
                        self.layers.append(leaky_relu)
                index += 1
            rnd_pic = np.random.uniform(-0.05, 0.05, picture_dim[0] * picture_dim[1] * picture_dim[2]).reshape(
                picture_dim[0], picture_dim[1], picture_dim[2])
            res = self.predict(rnd_pic, True)
            a, b, n, c = res.shape
            self._num_output = n * c
            self._num_train = num_train
            self.deep = NeuroBase.EvolvableNetwork(layer_nodes=deep_layers, num_input=self._num_output,
                                                   num_output=num_output, activation_function=deep_activation,
                                                   output_activation=deep_out, initialize=True)

        def predict(self, image, temp=False):
            image = image[..., np.newaxis].T
            for i in range(0, len(self.layers)):
                layer = self.layers[i]
                if i == 0:
                    result = layer.forward(image)
                else:
                    if isinstance(layer, AvgPool) or isinstance(layer, MaxPool) or isinstance(layer, Conv):
                        result = layer.forward(result)
                    else:
                        result = layer(result)
            if temp:
                return result
            return self.deep.predict(result.flatten())

    def _mutation_lognormal_cnn(self, par):
        child = copy.deepcopy(par)
        for i in range(0, par.deep.layer_count + 1):
            n, c = child.deep.layer_weights[i].shape
            child.deep.sigmas[i] += np.random.uniform(-0.01 * child.deep.sigmas[i], 0.01 * child.deep.sigmas[i], 1)[0]
            child.deep.layer_weights[i] += np.random.uniform(-child.deep.sigmas[i], child.deep.sigmas[i], n * c).reshape(n, c)
            if i == par.deep.layer_count:
                child.deep.layer_biases[i] += np.random.uniform(-child.deep.sigmas[i], child.deep.sigmas[i], 1)
            else:
                child.deep.layer_biases[i] += np.random.uniform(-child.deep.sigmas[i], child.deep.sigmas[i], c)
        index = 0
        for i in range(0, len(child.layers)):
            layer = child.layers[i]
            if isinstance(layer, Conv):
                child.sigmas[index] += np.random.uniform(-0.01*child.sigmas[index], 0.01*child.sigmas[index], 1)[0]
                layer.W['val'] += np.random.uniform(-child.sigmas[index], child.sigmas[index],
                                             layer.n_F*layer.n_C*layer.f*layer.f).reshape(layer.n_F, layer.n_C, layer.f,
                                                                                          layer.f)
                layer.b['val'] += np.random.uniform(-child.sigmas[index], child.sigmas[index], layer.n_F)
                index += 1
        return child

    def _algorithm_cnn(self, gen, fit, reinforcement):

        offspring_gen = []
        n = len(gen)
        for i in range(0, n):
            offspring_gen.append(self._mutation_lognormal_cnn(gen[i]))

        offspring_fit = reinforcement(offspring_gen)
        ind = np.asarray(range(0, 2 * n))
        temp = np.concatenate([fit, offspring_fit])
        ind = ind[np.argsort(-temp)]
        total = gen + offspring_gen
        return [total[i] for i in ind[0:n]], [temp[i] for i in ind[0:n]]

    def _initialize_networks_cnn(self, speciation=None):
        init_gen = []
        for i in range(0, self._max_gen_size):
            if speciation:
                num_layers = len(self.layer_nodes)
                activations = np.random.choice(speciation, num_layers).tolist()
                obj = self.EvolvableCNN(cnn_layers=self.cnn_layers,picture_dim=self.picture_dim,deep_layers=self.layer_nodes,
                                        num_output=self.num_output,
                                        deep_out=self.output_activation_name, deep_activation=activations)
            else:
                obj = self.EvolvableCNN(cnn_layers=self.cnn_layers, picture_dim=self.picture_dim, deep_layers=self.layer_nodes,
                                        num_output=self.num_output,
                                        deep_out=self.output_activation_name,
                                        deep_activation=self.activation_function_name)
            init_gen.append(obj)
        return init_gen

    def reinforce(self, max_epoch, speciation=None, verbose=True, warm_start=False, algorithm='speciation'):

        if warm_start:
            gen = self.last_gen
            max_epoch += self.prev_epoch
            species = self.species
            species_names = self.species_names
            fit = self._fit
        elif algorithm == 'speciation':
            species, species_names = self._create_species(self.species_layers)
            gen = self._initialize_networks_cnn(self.species_layers)
        else:
            gen = self._initialize_networks_cnn()
        if verbose:
            num_param = gen[0]._num_output * self.layer_nodes[0] + self.layer_nodes[0]
            for i in range(1, len(self.layer_nodes)):
                num_param += self.layer_nodes[i - 1] * self.layer_nodes[i] + self.layer_nodes[i]
            num_param += self.num_output * self.layer_nodes[len(self.layer_nodes) - 1] + 1
            if not warm_start:
                msg = "Number of Trainable Parameters For CNN Network: {}\n" \
                      "Number of Trainable Parameters for Deep Network: {}".format(gen[0]._num_train, num_param)
                print(msg)
        if not warm_start:
            fit = self._error_function(gen)
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
                gen, fit = self._algorithm_cnn(gen, fit, self._error_function)
            elif algorithm == 'speciation':
                gen, fit = self._algorithm_cnn(gen, fit, self._error_function)
                if verbose:
                    species_present = []
                    for i in range(0, len(gen)):
                        name = ""
                        for act in gen[i].deep.activation_function_name:
                            name = name + ',' + act
                        species_present.append(name[1:])
                    spec_count = []
                    species_present = np.asarray(species_present)
                    for spec in species_names:
                        spec_count.append(np.count_nonzero(np.where(species_present == spec)))
                    species_present = np.where(np.asarray(spec_count) > 0)[0]
                    # if reinforce -> check np.argmax in msg
                    msg = '  Number of Species Present for Deep: {}\n' \
                          '    Best Species by Top Fit: {}\n'.format(len(species_present),
                                                                     gen[np.argmax(fit)].deep.activation_function_name)
                    keys = list(species.keys())
                    for index in species_present:
                        msg += "    Species: [{}] Count: {}\n".format(keys[index], spec_count[index])
                    print(msg[:-1])  # skip last '\n'
        # fit = self._error_function(gen)
        self._fit = fit
        best_index = np.argmax(fit)
        self.best_model = gen[best_index]
        self.last_gen = gen
        self.prev_epoch = max_epoch
        if algorithm == 'speciation':
            self.species = species
            self.species_names = species_names

    def __init__(self, cnn_layers, deep_nodes, picture_dim, num_output, fitness_function, deep_activation='relu',
                 output_activation='softmax', population_size=20):
        if deep_activation is None:
            deep_activation = ['relu', 'tanh', 'sigmoid', 'gaussian', 'leaky_relu', 'selu']
        self.species_layers = deep_activation
        self._fit = None
        self._max_gen_size = population_size
        self.num_output = num_output
        self.layers = []
        self.cnn_activations = []
        self.cnn_layers = cnn_layers
        self.layer_nodes = deep_nodes
        self.output_activation_name = output_activation
        self.activation_function_name = deep_activation
        self.picture_dim = picture_dim
        self._error_function = fitness_function
        self.best_fit = []
        self.mean_fit = []
        self.last_gen = []
        self.best_model = []
        self.prev_epoch = 0
        self.species = None
        self.species_names = None
        self._fit = None

    def cnn_predict(self, image):
        image = image[..., np.newaxis].T
        for i in range(0, len(self.layers)):
            layer = self.layers[i]
            if i == 0:
                result = layer.forward(image)
            else:
                if isinstance(layer, AvgPool) or isinstance(layer, MaxPool) or isinstance(layer, Conv):
                    result = layer.forward(result)
                else:
                    result = layer(result)
        return result

    def predict(self, image):
        res = self.cnn_predict(image)
        res = res.flatten()





from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time

def fitness_function(gen, print_out=False):
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    fit = []
    max_game_count = 1
    max_step = 2000
    frame_count = 3
    max_x_pos = 125
    for ind in gen:
        score = []
        for j in range(0, max_game_count):
            local_score = 0
            next_state = env.reset()
            index_reset = 0
            accum = []
            prev_x = 40
            prev_count = 0
            for k in range(0, max_step):

                if print_out:
                    env.render()
                    print(k)
                    # time.sleep(0.015)
                if index_reset == 0:
                    index_reset = frame_count
                    if k == 0:
                        action = np.argmax(ind.predict(next_state))
                    else:
                        action = np.argmax(ind.predict(np.mean(accum, axis=0)))
                else:
                    index_reset -= 1
                next_state, reward, done, info = env.step(action)  # action.reshape(4,)
                if k > 65:
                    if info['x_pos'] <= 45:
                        #print("here")
                        #print(k)
                        #print(info['x_pos'])
                        done = True
                    if info['x_pos'] > (prev_x-1) and info['x_pos'] < (prev_x+1):
                        prev_count += 1
                        if prev_count == max_x_pos:
                            done = True
                    else:
                        prev_x = info['x_pos']
                local_score += reward
                accum.append(next_state)
                if done:
                    break
            score.append(local_score)
        fit.append(np.mean(score))
    return np.asarray(fit)

cnn_layers = []
cnn_layers.append("Conv(8,4)")  # (3*3*32)*RBG = 860
cnn_layers.append("tanh")
cnn_layers.append("AvgPool(2,2)")
cnn_layers.append("Conv(8, 4)")  # 3*3*32*prev_output = 9000
cnn_layers.append("sigmoid")
cnn_layers.append("MaxPool(2,2)")
cnn_layers.append("Conv(8,4)")  # 3*3*32*prev_output = 9000
cnn_layers.append("relu")
cnn_layers.append("AvgPool(2,2)")
cnn_layers.append("Conv(4,4)")  # 3*3*32*prev_output = 9000
cnn_layers.append("relu")
cnn_layers.append("AvgPool(2,2)")
cnn_layers.append("Conv(4,4)")  # 3*3*32*prev_output = 9000
cnn_layers.append("relu")
cnn_layers.append("MaxPool(2,2)")
cnn_layers.append("Conv(1,4)")  # 3*3*32*prev_output = 9000
cnn_layers.append("relu")
#model = NeuroReincorferImages(cnn_layers=cnn_layers, deep_nodes=[100, 100, 100], picture_dim=[256, 244, 3],num_output=7,
#                              fitness_function=fitness_function, deep_activation=['tanh', 'sigmoid', 'relu'])

#model.reinforce(max_epoch=10, speciation=None, verbose=True, warm_start=False)
import pickle
#pickle.dump(model, open("mario_model0", "wb"))

model = pickle.load(open("mario_model4", 'rb'))
#fitness_function([model.best_model], True)
max_epoch = 10
for i in range(5, 20):
    model.reinforce(max_epoch=max_epoch, verbose=True, warm_start=True)
    print("State Save")
    pickle.dump(model, open("mario_model{}".format(i), 'wb'))