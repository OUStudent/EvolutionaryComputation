from EvolutionaryComputation.util import *
from EvolutionaryComputation.AutoML.initial_populations import CustomInitialPopulation

class CustomAutoMLAlgorithm:
    """Generic Genetic Algorithm for Unconstrained Optimization Problems

        CustomAutoMLAlgorithm is a simple Genetic Algorithm that is designed
        for custom automated/hyper-parameter tuning machine learning algorithms.
        The backend genetic algorithm uses the same algorithms present
        in the GenericUnconstrainedProblem class. The main problem with
        using standard genetic algorithms when tuning machine learning models
        is that each fitness function is testing the parameters of a given
        model, which may take up to minutes. In this way, limiting the number
        of function calls is crucial to ensure an acceptable algorithm.

        The results of a genetic algorithm are directly correlated with the
        diversity of the initial population. Standard genetic algorithms only
        work with fixed sized populations. However, the CustomAutoMLAlgorithm
        utilizes a pre-existing CustomInitialPopulation from which to choose.
        In this way, the CustomInitialPopulation presents a large random
        sample of the possible domain space, representing diversity, while
        the CustomAutoMLAlgorithm employs exploitative genetic algorithms
        to converge and optimize the best handful of models during evolution.


        CustomAutoMLAlgorithm implements an "evolve" and "plot" method.

        Parameters
        -----------
        fitness_function : function pointer
            A pointer to a function that will evaluate and return the fitness of
            each individual in a population given their parameter values.
            Lastly, the function should return a numpy array of the fitness values.

        gen_size : int
            The number of individuals within each generation to perform evolution with.

        custom_initial_population : CustomInitialPopulation object
                                    A CustomInitialPopulation object that will act as the
                                    base initial population from which the evolution
                                    process will derive from. The best `gen_size` individuals
                                    from the CustomInitialPopulation object will be kept
                                    as the base generation for evolution.

        Attributes
        -----------
        gen : numpy 2D array
            A numpy 2D array of the individuals from the last generation of evolution.

        best_individual : numpy 1D array
            A numpy 1D array of the best individual from the last generation of evolution.

        best_fit : list
            A list of the best fitness values per generation of evolution.

        mean_fit : list
            A list of the mean fitness values per generation of evolution.

        best_values : list
            A list of the best individual per generation of evolution.

        """
    def __init__(self, fitness_function, custom_initial_population, gen_size, best_sigma=None):
        self.custom_initial_population = custom_initial_population
        self.gen_size = gen_size
        self.fitness_function = fitness_function
        self.best_fit = []
        self.mean_fit = []
        self.best_values = []
        self.upper_bound = custom_initial_population.upper_bound
        self.lower_bound = custom_initial_population.lower_bound
        self.total_bound = np.asarray(self.upper_bound) - np.asarray(self.lower_bound)
        self.domain = [self.upper_bound, self.lower_bound]
        self.num_variables = len(self.upper_bound)
        self.gen = None
        self.sigma = None
        self.best_individual = None
        self.best_sigma = best_sigma
        self.start_iter = 0

    # mutates the current generation to create offspring using
    # the strategy parameters sigma
    def __mutation_self_adaptive(self, parent, strategy):
        tau = 1 / (np.sqrt(2 * np.sqrt(self.num_variables)))
        tau_prime = 1 / (np.sqrt(2 * self.num_variables))
        r = np.random.normal(0, 1, len(parent))
        child_sigma = strategy * np.exp(tau * r + tau_prime * r)
        # r = np.random.laplace(0, 1, len(parent))
        r = np.random.normal(0, 1, len(parent))
        # r = np.random.standard_cauchy(len(parent))
        child_value = np.copy(parent) + child_sigma * r
        return child_value, child_sigma

    def __crossover_method_1(self, par):
        return np.mean(par, axis=0)

    def __crossover_method_2(self, par):
        child = np.copy(par[0])
        n = np.shape(par[0])[0]
        random_nums = np.random.randint(low=0, high=len(par), size=n)
        for j in range(0, n):
            child[j] = par[random_nums[j]][j]
        return child

    def __reproduction_self_adapt(self, par, sigma, f_par, find_max):
        c1_values, c1_sigma = self.__mutation_self_adaptive(par, sigma)
        c2_values, c2_sigma = self.__mutation_self_adaptive(par, sigma)
        c3_values, c3_sigma = self.__mutation_self_adaptive(par, sigma)
        c4_values, c4_sigma = self.__mutation_self_adaptive(par, sigma)

        total_val = np.asarray([c1_values, c2_values, c3_values, c4_values])
        total_sigma = np.asarray([c1_sigma, c2_sigma, c3_sigma, c4_sigma])
        for i in range(0, 4):
            for j in range(0, len(c1_values)):
                if total_val[i][j] > self.upper_bound[j]:
                    total_val[i][j] = self.upper_bound[j]
                    total_sigma[i][j] *= 0.9
                elif total_val[i][j] < self.lower_bound[j]:
                    total_val[i][j] = self.lower_bound[j]
                    total_sigma[i][j] *= 0.9

        f = self.fitness_function(total_val)
        total_val = np.vstack((par, total_val))
        total_sigma = np.vstack((sigma, total_sigma))
        f = np.asarray([f_par] + f.tolist())
        if find_max:
            bst = np.argmax(f)
            return total_val[bst], total_sigma[bst], f[bst]
        else:
            bst = np.argmin(f)
            return total_val[bst], total_sigma[bst], f[bst]

    def __reproduction_greedy(self, par, sigma, f_par, find_max):  # --- new arg ---
        c1_values = self.__crossover_method_1(par)
        c1_sigma = self.__crossover_method_1(sigma)
        c2_values = self.__crossover_method_1(par)
        c2_sigma = self.__crossover_method_1(sigma)
        c3_values, c3_sigma = self.__mutation_self_adaptive(c1_values, c1_sigma)
        c4_values, c4_sigma = self.__mutation_self_adaptive(c2_values, c2_sigma)
        c5_values = self.__crossover_method_2(par)
        c5_sigma = self.__crossover_method_2(sigma)
        c6_values = self.__crossover_method_2(par)
        c6_sigma = self.__crossover_method_2(sigma)
        c7_values, c7_sigma = self.__mutation_self_adaptive(c5_values, c5_sigma)
        c8_values, c8_sigma = self.__mutation_self_adaptive(c6_values, c6_sigma)

        total_val = [c1_values, c2_values, c3_values, c4_values, c5_values, c6_values, c7_values, c8_values]
        total_sigma = [c1_sigma, c2_sigma, c3_sigma, c4_sigma, c5_sigma, c6_sigma, c7_sigma, c8_sigma]

        for i in range(0, 8):
            for j in range(0, len(c1_values)):
                if total_val[i][j] > self.upper_bound[j]:
                    total_val[i][j] = self.upper_bound[j]
                    total_sigma[i][j] *= 0.9
                elif total_val[i][j] < self.lower_bound[j]:
                    total_val[i][j] = self.lower_bound[j]
                    total_sigma[i][j] *= 0.9

        f = self.fitness_function(np.asarray(total_val))
        total_val = par + total_val
        total_sigma = sigma + total_sigma
        f = f_par + f.tolist()
        if find_max:
            bst = np.argmax(f)
            return total_val[bst], total_sigma[bst], f[bst]
        else:
            bst = np.argmin(f)
            return total_val[bst], total_sigma[bst], f[bst]

    def __mutation_1_n_z(self, x1, xs, beta):
        return x1 + beta * (xs[0] - xs[1])

    def __differential(self, par, n, find_max, beta=0.5):
        ind = np.random.choice(range(0, n), 3, replace=False)
        target = self.gen[ind[2]]
        unit = self.__mutation_1_n_z(target, self.gen[ind[0:2]], beta)
        child = self.__crossover_method_1([unit, par])
        for j in range(0, len(child)):
            if child[j] > self.upper_bound[j]:
                child[j] = self.upper_bound[j]
            elif child[j] < self.lower_bound[j]:
                child[j] = self.lower_bound[j]
        for j in range(0, len(child)):
            if unit[j] > self.upper_bound[j]:
                unit[j] = self.upper_bound[j]
            elif unit[j] < self.lower_bound[j]:
                unit[j] = self.lower_bound[j]
        total = np.asarray([child])
        f = self.fitness_function(total)
        if find_max:
            bst = np.argmax(f)
            return f[bst], total[bst]
        else:
            bst = np.argmin(f)
            return f[bst], total[bst]

    def _create_pop_target(self, target, gen_size, mutation_percent,total_bounds):
        init_pop = np.empty(
            shape=(gen_size, len(target)))
        for i in range(0, gen_size):
            for j in range(0, len(target)):
                init_pop[i, j] = target[j] + np.random.uniform(-mutation_percent*total_bounds[j],
                                                               mutation_percent*total_bounds[j], 1)[0]
        return init_pop

    # greedy - proportional selection for mating - 4 cross 4 mut
    # differential
    def evolve(self, algorithm='differential', max_iter=100, info=True, find_max=False,
               warm_start=False):
        """Perform evolution with the given set of parameters

        Parameters
        -----------

        algorithm : string , default = 'greedy'
                   A string to denote the algorithm to be used for evolution.
                   Three algorithms are currently available: 'greedy', 'generic',
                   and 'self-adaptive'. Please see the example notebooks for a
                   run down.

        max_iter : int
                  The maximum number of iterations to run before terminating

        info : bool
              If True, print out information during the evolution process

        find_max : bool
                  If True, the algorithm will try to maximize the fitness
                  function given; else it will minimize.

        warm_start : bool
                    If True, the algorithm will use the last generation
                    from the previous generation instead of creating a
                    new initial population

        """
        if not warm_start:
            self.mean_fit = []
            self.best_fit = []

            if self.best_sigma is not None:
                if find_max:
                    bst = np.argsort(-self.custom_initial_population.fitness)[0]
                else:
                    bst = np.argsort(self.custom_initial_population.fitness)[0]

                target = np.copy(self.custom_initial_population.init_pop[bst])

                self.gen = self._create_pop_target(target, self.gen_size, self.best_sigma, self.total_bound)
                self.gen[0] = target
                if algorithm != 'differential':
                    init_sigma = np.empty(shape=(self.gen_size, self.num_variables))
                    for i in range(0, self.num_variables):
                        init_sigma[:, i] = np.random.uniform(0.01 * self.total_bound[i],
                                                             0.5*self.best_sigma * self.total_bound[i], self.gen_size)
                    self.sigma = np.copy(init_sigma)
                fitness = self.fitness_function(self.gen)

            else:
                if find_max:
                    bst = np.argsort(-self.custom_initial_population.fitness)[0:self.gen_size]
                else:
                    bst = np.argsort(self.custom_initial_population.fitness)[0:self.gen_size]

                self.gen = np.copy(self.custom_initial_population.init_pop[bst])

                if algorithm != 'differential':
                    init_sigma = np.empty(shape=(self.gen_size, self.num_variables))
                    for i in range(0, self.num_variables):
                        init_sigma[:, i] = np.random.uniform(0.01 * self.total_bound[i],
                                                             0.2 * self.total_bound[i], self.gen_size)
                    self.sigma = np.copy(init_sigma)
                fitness = np.copy(self.custom_initial_population.fitness[bst])
        else:

            fitness = self.fitness_function(self.gen)

        n, c = np.shape(self.gen)
        for k in range(self.start_iter, self.start_iter+max_iter):
            fit_mean = np.mean(fitness)
            if find_max:
                fit_best = np.max(fitness)
                best_index = np.argmax(fitness)
            else:
                fit_best = np.min(fitness)
                best_index = np.argmin(fitness)
            self.best_values.append(self.gen[best_index,])
            self.best_fit.append(fit_best)
            self.mean_fit.append(fit_mean)
            if info:
                msg = "GENERATION {}:\n" \
                      "  Best Fit: {}, Mean Fit: {}".format(k, fit_best, fit_mean)
                print(msg)

            if algorithm == 'differential':
                if find_max:
                    coef = fit_mean / fit_best
                else:
                    coef = fit_best / fit_mean
                if coef > 0.95:
                    beta = 1
                elif coef < 0.2:
                    beta = 0.2
                else:
                    beta = 0.55
                for i in range(0, n):
                    par = self.gen[i]
                    f, child = self.__differential(par, n, find_max, beta=beta)
                    if find_max:
                        if f > fitness[i]:
                            fitness[i] = f
                            self.gen[i] = child
                    else:
                        if f < fitness[i]:
                            fitness[i] = f
                            self.gen[i] = child
            elif algorithm == 'greedy':
                mates1 = np.random.choice(range(0, n), n, replace=False)
                mates2 = np.random.choice(range(0, n), n, replace=False)
                children = []
                children_sigma = []
                fits = []
                for i in range(0, n):
                    v, s, f = self.__reproduction_greedy([self.gen[mates1[i]], self.gen[mates2[i]]],
                                                         [self.sigma[mates1[i]], self.sigma[mates2[i]]],
                                                         [fitness[mates1[i]], fitness[mates2[i]]],
                                                         find_max)
                    children.append(v)
                    children_sigma.append(s)
                    fits.append(f)
                fitness = fits

            elif algorithm == 'self-adaptive':
                children = []
                children_sigma = []
                fits = []
                for i in range(0, n):
                    c_value, c_sigma, f = self.__reproduction_self_adapt(self.gen[i], self.sigma[i], fitness[i],
                                                                         find_max)
                    children.append(c_value)
                    children_sigma.append(c_sigma)
                    fits.append(f)
                fitness = fits

            if algorithm != 'differential':
                gen_next = np.asarray(children)
                sigma_next = np.asarray(children_sigma)

                self.gen = gen_next
                if algorithm == 'self-adaptive':
                    self.sigma = sigma_next

        if find_max:
            self.best_individual = self.gen[np.argmax(fitness)]
        else:
            self.best_individual = self.gen[np.argmin(fitness)]
        self.start_iter += max_iter

    def plot(self, starting_gen=0):
        """Plots the best and mean fitness values after the evolution process.

        Parameters
        -----------

        starting_gen : int
                      The starting index for plotting.
        """
        x_range = range(starting_gen, len(self.best_fit))
        plt.plot(x_range, self.mean_fit[starting_gen:], label="Mean Fitness")
        plt.plot(x_range, self.best_fit[starting_gen:], label="Best Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Value")
        plt.suptitle("Mean and Best Fitness for Algorithm: ")
        plt.legend()
        plt.show()


