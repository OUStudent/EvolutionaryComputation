from EvolutionaryComputation.util import *


def hyper_param_sample(module_probs, hyper_param_prob, num_sample=1000, plot=True):
    cnts = [0]*len(module_probs)
    hyper_param_prob = np.asarray(hyper_param_prob)

    for i in range(0, num_sample):
        r = np.random.uniform(0, 1, len(module_probs))
        r2 = np.random.uniform(0, 1, len(np.where(r <= module_probs)[0]))
        cnts[len(np.where(r2 <= hyper_param_prob[r <= module_probs])[0])-1] += 1
    temp = pd.DataFrame(np.vstack((np.asarray(range(1, len(module_probs) + 1)), np.asarray(cnts))))
    temp.index = ['Module Counts', "Num of Occurrences"]
    temp.columns = [''] * len(module_probs)
    if plot:
        plt.bar(range(1, len(cnts)+1), cnts)
        plt.ylabel("Number of Occurrences")
        plt.xlabel("Module Count")
        plt.suptitle("Hyper-parameter Sample")
        plt.show()
    return temp

class Saved_Weights:

    def __init__(self, weights, type, val_loss):
        self.type = type
        self.weights = weights
        self.val_loss = val_loss

class NetworkModule:
    def __init__(self, prob_include):
        self.layers = []
        self.hyperparam = []
        self.prob_include = prob_include


class CustomDeepModule(NetworkModule):

    def __init__(self, prob_include):
        super().__init__(prob_include)

    def addDense(self, min_node, max_node, activations=None, activation_probs=None):
        if activations is None:
            activations = ['relu', 'selu', 'elu', 'leaky_relu']
        if activation_probs is None:
            activation_probs = [0.25, 0.25, 0.25, 0.25]
        self.layers.append("dense")
        self.hyperparam.append([min_node, max_node, activations, activation_probs])

    def addBatchNorm(self, prob_include=None):
        self.layers.append("batchNorm")
        self.hyperparam.append([prob_include])

    def addDropOut(self, prob_include, min_alpha=0.2, max_alpha=0.5):
        self.layers.append("dropOut")
        self.hyperparam.append([prob_include, min_alpha, max_alpha])


class CustomConvolutionModule(NetworkModule):

    def __init__(self, prob_include):
        super().__init__(prob_include)

    def addConvolution(self, channel_sizes=None, ch_prob=None):
        if channel_sizes is None:
            channel_sizes = [32, 64, 128, 256]
        if ch_prob is None:
            ch_prob = [0.25, 0.25, 0.25, 0.25]
        self.layers.append("conv")
        self.hyperparam.append([channel_sizes, ch_prob])

    def addPooling(self, max_prob=0.5, avg_prob=0.5, prob_include=None):
        self.layers.append("pooling")
        self.hyperparam.append([prob_include, [avg_prob, max_prob]])

    def addBatchNorm(self, prob_include=None):
        self.layers.append("batchNorm")
        self.hyperparam.append([prob_include])

    def addActivation(self, activations, activation_probs, prob_include=None):
        self.layers.append("activation")
        if activations is None:
            activations = ['relu', 'selu', 'elu', 'leaky_relu']
        if activation_probs is None:
            activation_probs = [0.25, 0.25, 0.25, 0.25]
        self.hyperparam.append([prob_include, activations, activation_probs])

    def addDropOut(self, prob_include, min_alpha=0.2, max_alpha=0.5):
        self.layers.append("dropOut")
        self.hyperparam.append([prob_include, min_alpha, max_alpha])

