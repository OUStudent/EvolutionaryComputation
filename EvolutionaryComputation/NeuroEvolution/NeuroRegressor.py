from NeuroBase import NeuroBase
from EvolutionaryComputation.util import *

class NeuroRegressor(NeuroBase):

    def __init__(self, layer_nodes, num_input, num_output, activation_function="relu",
                 population_size=100):
        super().__init__(layer_nodes=layer_nodes, num_input=num_input, num_output=num_output,
                         activation_function=activation_function, output_activation='purlin',
                         error_function=mse_error, population_size=population_size)

    def evolve(self, max_epoch, batch_size, train_data, val_data, early_stopping=True,
               verbose=True, warm_start=False, algorithm='generic', patience=10):
        super().evolve(max_epoch=max_epoch, batch_size=batch_size, train_data=train_data, val_data=val_data,
                       early_stopping=early_stopping, verbose=verbose, warm_start=warm_start,
                       algorithm=algorithm, patience=patience)

    def predict(self, data):
        return self.best_model.predict(data)

    def plot(self, starting_gen=0):
        super().plot(self.mean_fit, self.best_fit, self.val_fit, starting_gen=starting_gen)