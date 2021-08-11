from EvolutionaryComputation.util import *

class NetworkModule:
    def __init__(self, prob_include):
        self.layers = []
        self.hyperparam = []
        self.prob_include = prob_include


class CustomDeepModule(NetworkModule):

    def __init__(self, prob_include):
        super().__init__(prob_include)

    def addDense(self, min_node, max_node, activation):
        self.layers.append("dense")
        self.hyperparam.append([min_node, max_node, activation])

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

    def addActivation(self, activation, prob_include=None):
        self.layers.append("activation")
        self.hyperparam.append([prob_include, activation])

    def addDropOut(self, prob_include, min_alpha=0.2, max_alpha=0.5):
        self.layers.append("dropOut")
        self.hyperparam.append([prob_include, min_alpha, max_alpha])

