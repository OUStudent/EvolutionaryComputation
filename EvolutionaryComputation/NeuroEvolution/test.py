import EvolutionaryComputation.NeuroEvolution as ne
import numpy as np
import gym
import pickle
from matplotlib import animation
import matplotlib.pyplot as plt
from PIL import Image


def fitness_function(gen, print_out=False):
    env = gym.make("Pong-ram-v0")
    fit = []
    max_game_count = 1
    max_step = 250
    index = 0
    for ind in gen:
        score = []
        frames = []
        for j in range(0, max_game_count):
            local_score = 0
            next_state = env.reset()
            for k in range(0, max_step):
                if print_out:
                    #frames.append(Image.fromarray(env.render(mode='rgb_array')))
                    env.render()
                    print(k)
                    plt.pause(1/60)
                action = np.argmax(ind.predict(next_state))
                next_state, reward, done, info = env.step(action)
                local_score += reward
                if done:
                    break
            score.append(local_score)
            '''
            with open('best{}.gif'.format(index), 'wb') as f:  # change the path if necessary
                im = Image.new('RGB', frames[0].size)
                im.save(f, save_all=True, append_images=frames, duration=25, loop=0)
            index += 1
            '''
        fit.append(score[0])
    return np.asarray(fit)

#model = pickle.load(open("model300", "rb"))
#fitness_function([model.best_model], print_out=True)

layer_nodes = [50, 50, 50]
num_input = 128
num_output = 6
fitness_function = fitness_function
population_size = 50
output_activation = 'softmax'
activation_function = ['selu', 'tanh', 'elu', 'relu', 'gaussian', 'sigmoid']
max_epochs = 100

model = ne.NeuroReinforcer(layer_nodes=layer_nodes, num_input=num_input, num_output=num_output,
                          fitness_function=fitness_function, population_size=population_size,
                          output_activation=output_activation,
                          activation_function=activation_function)

model.evolve(max_epoch=max_epochs, verbose=True, warm_start=False,
             just_layers=True, algorithm='speciation', prob_chnge_species=0.10)



#print(fitness_function([model.best_model], print_out=True))
#print(fitness_function([model.best_model], print_out=True))
#print(fitness_function([model.best_model], print_out=True))
print(fitness_function([model.best_model]))
print(fitness_function(model.last_gen))
print(fitness_function(model.last_gen))
pickle.dump(model, open("model100", "wb"))
model.plot()
model.plot(starting_gen=50)
model.plot(plot_species=True)


