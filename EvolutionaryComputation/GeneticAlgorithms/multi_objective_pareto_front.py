from EvolutionaryComputation.util import *


class ParetoFrontMOP:

    def __init__(self, fitness_functions, upper_bound, lower_bound, gen_size):
        self.gen_size = gen_size
        self.fitness_functions = fitness_functions
        self.num_variables = len(upper_bound)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.total_bound = np.asarray(upper_bound) - np.asarray(lower_bound)
        self.domain = [upper_bound, lower_bound]
        self.gen = None
        self.sigma = None
        self.best_individual = None
        self.tourn_size = int(0.1 * self.gen_size)
        self.archive = []

    def __distance(self, x):
        res = []
        for j in range(0, len(x)):
            res.append(np.linalg.norm(x[j] - x[:, ]))
        return res

    # return the relative fitness of an x value
    # compared to the tournament x values: tourn_x,
    # based off pareto dominance. Reward is based
    # off strong and weak domination
    def __pareto_dominance(self, x, tourn_x):
        fx = self.fitness_functions(x)
        tourn_f = self.fitness_functions(tourn_x)
        score = 0
        if len(tourn_x.shape) == 1:
            if np.sum(fx <= tourn_f) == len(fx):  # atleast equal in all functions
                score += 2
                if np.sum(fx < tourn_f) >= 1:  # better in one function so strong dominance
                    score += 2
            elif np.sum(fx < tourn_f) >= 1:  # better in one function but not equal in rest, so weak dominance
                score += 2
        else:
            for i in range(0, len(tourn_x)):
                if np.sum(fx <= tourn_f[i]) == len(fx):  # atleast equal in all functions
                    score += 2
                    if np.sum(fx < tourn_f[i]) >= 1:  # better in one function so strong dominance
                        score += 2
                elif np.sum(fx < tourn_f[i]) >= 1:  # better in one function but not equal in rest, so weak dominance
                    score += 2
        return score

    # Handles more than 2 parents
    # Averaging Technique by taking mean of all parent values
    def __crossover_method_1(self, par):
        child = np.copy(par[0])
        for j in range(0, np.shape(par[0])[0]):
            child[j] = np.mean(par[:, j])
        return child

    # Handles more than 2 parents
    # 'intuitive' crossover by uniformly randomly
    # inheriting genes from parents
    def __crossover_method_2(self, par):
        child = np.copy(par[0])
        n = np.shape(par[0])[0]
        # random indices from parents for child to inherit from
        random_nums = np.random.randint(low=0, high=len(par), size=n)
        for j in range(0, n):
            child[j] = par[random_nums[j]][j]
        return child

    # classical Evolutionary Programming Mutation
    # log-normal self adaptation
    def __mutate(self, par_val, par_sigma):
        tau = 1 / (np.sqrt(2 * np.sqrt(self.num_variables)))
        tau_prime = 1 / (np.sqrt(2 * self.num_variables))
        r = np.random.normal(0, 1, len(par_sigma))
        ch_sigma = par_sigma * np.exp(tau * r + tau_prime * r)
        r = np.random.normal(0, 1, len(par_sigma))
        ch_val = np.copy(par_val) + ch_sigma * r
        return ch_val, ch_sigma

    # given a set of parents, their sigma, and cross_method
    # produce two offspring, one through crossover only
    # and the other through mutation after crossover
    # choose the best for survival based off pareto-dominance
    def __reproduction(self, par, sigma, cross_method):
        if cross_method == 1:
            child_val = self.__crossover_method_1(par)
            child_sigma = self.__crossover_method_1(sigma)
        else:
            child_val = self.__crossover_method_2(par)
            child_sigma = self.__crossover_method_2(sigma)
        # mutate crossover offspring
        mut_child_val, mut_child_sigma = self.__mutate(child_val, child_sigma)
        # get pareto-dominance between offspring
        f = self.__pareto_dominance(child_val, mut_child_val)
        if f == 4:  # strong domination
            return child_val, child_sigma
        elif f == 2:  # weak domination
            r = np.random.uniform(0, 1, 1)
            if r < 0.5:  # randomly select winner
                return child_val, child_sigma
            else:
                return mut_child_val, mut_child_sigma
        else:  # no domination
            return mut_child_val, mut_child_sigma

    # function has been parameterized for our problem, so no initial gen given
    # size is # of individuals, var_count is number of variables for our problem
    # max_distance is max euclidean similarity distance allowed for solutions,
    # max_iter is max iter of inner loop, archive_iter is outer loop -> larger
    # iter for archive creates a bigger pareto-front
    def evolve(self, max_iter=40, warm_start=False, sigma_bound=None, info=True, cross_method=1, par_count=3,
               max_distance=2.0, archive_iter=30, max_archive_size=200):

        if sigma_bound is None:
            sigma_bound = [0.01, 0.2]

        if not warm_start:
            self.archive = []
        else:
            self.archive = self.archive.tolist()

        # outer loop
        for q in range(0, archive_iter):
            # create initial generation with strategy parameters
            init_gen = np.empty(shape=(self.gen_size, self.num_variables))
            init_sigma = np.empty(shape=(self.gen_size, self.num_variables))
            for i in range(0, self.num_variables):
                init_gen[:, i] = np.random.uniform(self.lower_bound[i], self.upper_bound[i], self.gen_size)
                init_sigma[:, i] = np.random.uniform(sigma_bound[0] * self.total_bound[i],
                                                     sigma_bound[1] * self.total_bound[i], self.gen_size)

            self.gen = np.copy(init_gen)
            self.sigma = np.copy(init_sigma)

            if info:
                msg = " Archive Iteration: {}/{}".format(q+1, archive_iter)
                print(msg)
            self.gen = np.copy(init_gen)
            self.sigma = np.copy(init_sigma)

            # inner loop
            for k in range(0, max_iter):

                if info:
                    if k == max_iter - 1:
                        msg = "Generation {}/{}\n".format(k+1, max_iter)
                    else:
                        msg = "Generation {}/{}".format(k+1, max_iter)
                    sys.stdout.write("\r"+msg)

                # random selection based off number of parents
                selected = []
                for i in range(0, par_count):
                    selected.append(np.random.choice(range(0, self.gen_size), self.gen_size, replace=False))
                selected = np.asarray(selected).T
                ch_val = []
                ch_sigma = []
                # reproduce from parents
                for i in range(0, self.gen_size):
                    c_v, c_s = self.__reproduction(self.gen[selected[i]], self.sigma[selected[i]], cross_method)
                    ch_val.append(c_v)
                    ch_sigma.append(c_s)

                ch_val = np.asarray(ch_val)
                ch_sigma = np.asarray(ch_sigma)
                # check bounds of offspring
                for i in range(0, self.gen_size):
                    for j in range(0, self.num_variables):
                        if ch_val[i][j] > self.domain[0][j]:
                            ch_val[i][j] = self.domain[0][j]
                            ch_sigma[i][j] *= .90
                        elif ch_val[i][j] < self.domain[1][j]:
                            ch_val[i][j] = self.domain[1][j]
                            ch_sigma[i][j] *= .90

                # combine offspring with parents
                parents_offspring_val = np.vstack((self.gen, ch_val))
                parents_offspring_sigma = np.vstack((self.sigma, ch_sigma))
                ind = list(range(0, 2 * self.gen_size))
                rel_fit = []
                # compute relative fitness
                for i in range(0, 2 * self.gen_size):
                    tourn = np.random.choice(ind[0:i] + ind[(i + 1):], self.tourn_size)
                    rel_fit.append(self.__pareto_dominance(parents_offspring_val[i], parents_offspring_val[tourn]))

                if len(self.archive) != 0:  # if archive is not emtpy
                    # loop over each member in archive
                    for arch in self.archive:
                        for i in range(0, 2 * self.gen_size):
                            # calculate euclidean distance
                            dist = np.linalg.norm(arch - parents_offspring_val[i])
                            # if their distance is small -> they're too close,
                            # larger the max distance the more spread out solutions become
                            if dist <= max_distance:
                                # penalize fitness down to 25%
                                rel_fit[i] *= 0.25

                # sort individuals based on relative fitness
                rel_fit = np.asarray(rel_fit)
                sorted_ind = np.asarray(ind)[np.argsort(-rel_fit)]

                # use elitism to take the best half from pooled
                # parents and offspring
                self.gen = parents_offspring_val[sorted_ind[0:self.gen_size]]
                self.sigma = parents_offspring_sigma[sorted_ind[0:self.gen_size]]

            fits = self.fitness_functions(self.gen)
            # update archive
            if len(self.archive) !=0:
                # loop over every individual in surviving generation
                for j in range(0, self.gen_size):
                    # indices of members in archive that are dominated by
                    # the current individual
                    ind_del = []
                    f_arch = self.fitness_functions(np.asarray(self.archive))
                    dominated = False
                    for i in range(0, len(self.archive)):
                        # individual strongly dominates member of archive
                        if np.sum(f_arch[i] > fits[j]) == len(fits[j]):
                            ind_del.append(i)
                        # individual is strongly dominated by member
                        elif np.sum(f_arch[i] < fits[j]) == len(fits[j]):
                            dominated = True
                    if len(ind_del) != 0:
                        # delete members that are strongly dominated
                        for index in sorted(ind_del, reverse=True):
                            del self.archive[index]
                    if not dominated:  # weakly dominates members of archive
                        if len(self.archive) <= max_archive_size:
                            self.archive.append(self.gen[j])
                        else:  # archive is full
                            temp = np.vstack((self.gen[j], self.archive))
                            dist = self.__distance(temp)
                            index = np.argmin(dist)
                            if index != 0:
                                del self.archive[index-1]
                                self.archive.append(self.gen[j])

                dist = pdist(np.asarray(self.archive))
                #dist = self.__distance(np.asarray(self.archive))
                dist_min = np.min(dist)
                dist_max = np.max(dist)
                dist_median = np.median(dist)
                dist_std = np.std(dist)
                if info:
                    msg = "   Archive Information:\n" \
                          "   Distance - Min: {}, Max: {}, Median: {}, Std: {}\n" \
                          "   Archive Size: {}".format(dist_min, dist_max, dist_median,dist_std, len(self.archive))
                    print(msg)

            else:  # if first iteration of outer loop, just add best solution from gen
                self.archive.append(self.gen[0])
        self.archive = np.asarray(self.archive)

    def plot(self):
        fx = self.fitness_functions(self.archive)
        n = len(fx[0])

        if n == 2:
            plt.xlabel("F1")
            plt.ylabel("F2")
            plt.suptitle("Pareto Front")
            plt.scatter(fx[:,0], fx[:,1], label='Archive')
            plt.show()
        elif n == 3:
            plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter(fx[:, 0], fx[:, 1], fx[:, 2])
            ax.set_xlabel("F1")
            ax.set_ylabel("F2")
            ax.set_zlabel("F3")
            plt.suptitle("Pareto Front of Archive")
            plt.show()
        else:
            print("Cannot Print Multi-Dimensional Front greater than 3D")



