from EvolutionaryComputation.util import *


class ConstrainedProblem:
    """Genetic Algorithm for Constrained Optimization Problems

        ConstrainedProblem is an algorithm that works by utilizing relative fitness
        instead of raw fitness. Relative fitness is computed by examining how many
        times the current individual's raw fitness is better than a random selection
        of individuals from the population. The main problem concerning constrained
        problems is dealing with solutions that break the constraints. To deal with
        this, the algorithm works by splitting the solutions into two groups: those
        that violate the constraints, and those that do not. When computing relative
        fitness, if two solutions are compared that do not break constraints, the
        solution with the better raw fitness is chosen. If two solutions are chosen
        where one breaks a constraint and the other does not, the solution that does
        not break any of the constraints is chosen. Lastly, if two solutions are
        chosen where they both break the constraints, the solution whose constraints
        are broken 'less' is chosen.

        ConstrainedProblem implements an "evolve" and "plot" method.

        Parameters
        -----------
        fitness_function : function pointer
            A pointer to a function that will evaluate and return the fitness of
            each individual in a population given their parameter values. Lastly, the
            function should return a numpy array of the fitness values.

        constraints : function pointer
            A pointer to a function that will evaluate all the constraints for
            each individual in a population given their parameter value. Lastly, the
            function should return a numpy array of the fitness values.

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

        gen_size : int
            The number of individuals within each generation to perform evolution with.

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

        """

    def __init__(self, fitness_function, constraints, upper_bound, lower_bound, gen_size):
        self.gen_size = gen_size
        self.fitness_function = fitness_function
        self.constraints = constraints
        self.best_fit = []
        self.mean_fit = []
        self.mean_sigmas = []
        self.num_variables = len(upper_bound)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.total_bound = np.asarray(upper_bound) - np.asarray(lower_bound)
        self.domain = [upper_bound, lower_bound]
        self.gen = None
        self.sigma = None
        self.best_individual = None
        self.tourn_size = int(0.2 * self.gen_size)

    def __constraints(self, x):
        g = self.constraints(x)
        res = []
        for ind in g:
            t_res = [0]
            for con in ind:
                if con > 0:
                    t_res.append(con)
                    t_res[0] += 1
                else:
                    t_res.append(0)
            res.append(t_res)
        return np.asarray(res)

    # fit is current individual's fitness
    # tourn_fit is the fitness from the tournament
    def __calculate_relative_fit(self, fit, tourn_fit, find_max):
        s = 0
        r = np.random.uniform(0, 1, len(tourn_fit))
        for i in range(0, len(tourn_fit)):
            if fit == tourn_fit[i]:  # if fitness is equal, 50/50 chance of being better
                if r[i] >= 0.5:
                    s += 1
            else:
                if find_max:
                    if fit > tourn_fit[i]:
                        s += 1
                else:
                    if fit < tourn_fit[i]:
                        s += 1
        return s

    # mutates the current generation to create offspring using
    # the strategy parameters sigma
    def __mutation(self):
        tau = 1 / (np.sqrt(2 * np.sqrt(self.num_variables)))
        tau_prime = 1 / (np.sqrt(2 * self.num_variables))
        offspring_values = []
        offspring_sigma = []
        # loop over each individual and associated strategy parameters
        for (parent, strategy) in zip(self.gen, self.sigma):
            r = np.random.normal(0, 1, len(parent))
            child_sigma = strategy * np.exp(tau * r + tau_prime * r)

            # r = np.random.laplace(0, 1, len(parent))
            r = np.random.normal(0, 1, len(parent))
            # r = np.random.standard_cauchy(len(parent))
            child_value = np.copy(parent) + child_sigma * r

            offspring_values.append(child_value)
            offspring_sigma.append(child_sigma)
        return np.asarray(offspring_values), np.asarray(offspring_sigma)

    def plot(self, plot_sigmas=False, starting_gen=0):
        """Plots the best and mean fitness values after the evolution process.

        Parameters
        -----------

        starting_gen : int
                      The starting index for plotting.

        plot_sigmas : bool
                     If true, the self-adaptive mutation variations will
                     be plotted as well.
        """
        x_range = range(starting_gen, len(self.best_fit))
        if plot_sigmas:
            mean_sigmas = np.asarray(self.mean_sigmas[starting_gen:,])
            for i in reversed(range(starting_gen, self.sigma.shape[1])):
                plt.figure(2 + i)
                plt.plot(x_range, mean_sigmas[:, i])
                plt.xlabel("Generation")
                plt.ylabel("Strategy Parameter Value")
                plt.suptitle("Mean Strategy Parameter {}".format(i + 1))
        plt.figure(1)
        plt.plot(x_range, self.mean_fit[starting_gen:], label="Mean Fitness")
        plt.plot(x_range, self.best_fit[starting_gen:], label="Best Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Value")
        plt.suptitle("Mean and Best Fitness")
        plt.legend()
        plt.show()

    def evolve(self, max_iter=300, warm_start=False, sigma_bound=None, find_max=False, info=True):
        """Perform evolution with the given set of parameters

        Parameters
        -----------

        max_iter : int
                  The maximum number of iterations to run before terminating

        info : bool
              If True, print out information during the evolution process

        warm_start : bool
                    If True, the algorithm will use the last generation
                    from the previous generation instead of creating a
                    new initial population

        sigma_bound : bool or two value list
                    If None, defaults to ``[0.01, 0.2]``. This represents
                    the mutation bounds for mutation. First index represents
                    the lower bound while the second index represents the upper
                    bound.

        find_max : bool
                  If True, the algorithm will try to maximize the fitness
                  function given; else it will minimize.

        """
        if sigma_bound is None:
            sigma_bound = [0.01, 0.2]

        if not warm_start:
            self.mean_fit = []
            self.best_fit = []
            init_gen = np.empty(shape=(self.gen_size, self.num_variables))
            init_sigma = np.empty(shape=(self.gen_size, self.num_variables))
            for i in range(0, self.num_variables):
                init_gen[:, i] = np.random.uniform(self.lower_bound[i], self.upper_bound[i], self.gen_size)
                init_sigma[:, i] = np.random.uniform(sigma_bound[0] * self.total_bound[i],
                                                     sigma_bound[1] * self.total_bound[i], self.gen_size)

            self.gen = np.copy(init_gen)
            self.sigma = np.copy(init_sigma)

        for k in range(0, max_iter):

            offspring_values, offspring_sigma = self.__mutation()
            # loop over all individuals
            for i in range(0, len(offspring_values)):
                # loop over all columns/variables
                for j in range(0, len(self.domain[0])):
                    if offspring_values[i][j] > self.domain[0][j]:
                        offspring_values[i][j] = self.domain[0][j]
                        offspring_sigma[i][j] *= .90
                    elif offspring_values[i][j] < self.domain[1][j]:
                        offspring_values[i][j] = self.domain[1][j]
                        offspring_sigma[i][j] *= .90

            parents_offspring_val = np.vstack((self.gen, offspring_values))
            parents_offspring_sigma = np.vstack((self.sigma, offspring_sigma))

            fit = self.fitness_function(parents_offspring_val)
            # constraint results
            con = self.__constraints(parents_offspring_val)
            # first index of each row above ^ returns how many constraints
            # are broken if it is 0, no constraints broken so we get all
            # indices of the good individuals
            good = np.where(con[:, 0] == 0)[0].tolist()
            if len(good) >= 1:
                if find_max:
                    fit_best = np.max(fit[good])
                else:
                    fit_best = np.min(fit[good])
                fit_mean = np.mean(fit[good])
                sigma_means = []
                for i in range(0, self.sigma.shape[1]):
                    sigma_means.append(np.mean(self.sigma[:, i]))
                self.mean_sigmas.append(sigma_means)
                self.best_fit.append(fit_best)
                self.mean_fit.append(fit_mean)

            if info:
                msg = "GENERATION {}:\n" \
                      "  Best Fit: {}, Mean Fit: {}".format(k, fit_best, fit_mean)
                if len(good) < self.gen_size:
                    msg = "GENERATION {}:\n" \
                          "  No Feasible Solutions in current Generation".format(k)
                print(msg)

            rel_fit = []
            # loop over each good solution
            if len(good) > 1:
                for i in range(0, len(good)):
                    # tournament indices
                    tourn = np.random.choice(good[0:i] + good[(i + 1):], self.tourn_size)
                    rel_fit.append(self.__calculate_relative_fit(fit[good[i]], fit[tourn], find_max))
                # now sort indices based off relative fitness in decreasing value
                good = np.asarray(good)[np.argsort(-np.asarray(rel_fit))]

            # get indices of bad solutions, aka when number of constraints broken
            # is not equal to 0
            bad = np.where(con[:, 0] != 0)[0].tolist()
            # if has constraints broken, we returned the difference by how much
            # it was broken, so here we get those differences
            violated_con = con[bad][:, 1:]
            # now we calculate the z score by using [x-mean(x)]/std(x)
            z_scores = (violated_con - np.mean(violated_con, axis=0)) / (np.std(violated_con, axis=0))
            # if an entire column for a constraint is the same then it
            # would result in nan values, so we replace with 0
            z_scores[np.isnan(z_scores)] = 0
            # sum up the z score for each constraint
            scores = np.sum(z_scores, axis=1)
            # individuals with a larger sum indicate a larger difference broken
            bad = np.asarray(bad)[np.argsort(scores)]

            next_gen_val = np.empty(shape=(self.gen_size, self.num_variables))
            next_gen_sigma = np.empty(shape=(self.gen_size, self.num_variables))
            # there are more good solutions than needed for next generation
            if len(good) >= self.gen_size:
                next_gen_val = parents_offspring_val[good[0:self.gen_size]]
                next_gen_sigma = parents_offspring_sigma[good[0:self.gen_size]]
            elif len(good) == 0:  # no feasible solutions
                next_gen_val = parents_offspring_val[bad[0:self.gen_size]]
                next_gen_sigma = parents_offspring_sigma[bad[0:self.gen_size]]
            else:
                # not enough good solutions so we take all good and the
                # rest are the best bad solutions
                diff = self.gen_size - len(good)
                next_gen_val[0:len(good)] = parents_offspring_val[good]
                next_gen_sigma[0:len(good)] = parents_offspring_sigma[good]
                next_gen_val[len(good):self.gen_size] = parents_offspring_val[bad[0:diff]]
                next_gen_sigma[len(good):self.gen_size] = parents_offspring_sigma[bad[0:diff]]
            self.gen = next_gen_val
            self.sigma = next_gen_sigma

        self.best_individual = self.gen[0]
        if info:
            msg = "--- MAXIMUM NUMBER OF ITERATIONS REACHED---\n" \
                  " Best Individual: "
            print(msg)
            print(self.best_individual)



