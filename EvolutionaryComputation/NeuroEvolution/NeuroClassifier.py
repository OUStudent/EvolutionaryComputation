from NeuroBase import NeuroBase
from EvolutionaryComputation.util import *

class NeuroClassifier(NeuroBase):

    def __init__(self, layer_nodes, num_input, num_output, activation_function="relu",
                 population_size=100, multi_label=False):
        self.multi_label = multi_label
        if num_output == 1:
            self.binary = True
        else:
            self.binary = False
        if num_output == 1:  # binary class
            super().__init__(layer_nodes=layer_nodes, num_input=num_input, num_output=num_output,
                             activation_function=activation_function, output_activation='logistic',
                             error_function=cross_entropy_error, population_size=population_size)
        else:  # multi-class or multi-label
            super().__init__(layer_nodes=layer_nodes, num_input=num_input, num_output=num_output,
                             activation_function=activation_function, output_activation='softmax',
                             error_function=cross_entropy_error, population_size=population_size)

    def evolve(self, max_epoch, batch_size, train_data, val_data, early_stopping=True,
               verbose=True, warm_start=False, algorithm='generic', patience=10):
        super().evolve(max_epoch=max_epoch, batch_size=batch_size, train_data=train_data, val_data=val_data,
                       early_stopping=early_stopping, verbose=verbose, warm_start=warm_start,
                       algorithm=algorithm, patience=patience)

    def predict(self, data):
        temp = self.best_model.predict(data)
        if self.multi_label or self.binary:
            temp = np.array(temp > 0.5, dtype=int)
        else:
            temp = np.argmax(temp, axis=1)
        return temp

    def plot(self):
        super().plot(self.mean_fit, self.best_fit, self.val_fit)
