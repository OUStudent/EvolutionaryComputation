
from EvolutionaryComputation.util import *


class HyperParamUnconstrainedProblem:

    def __init__(self, fitness_function, upper_bound, lower_bound, gen_size, min_prob=0.05, min_mut=0.001):
        self.gen_size = gen_size
        self.fitness_function = fitness_function
        self.best_fit = []
        self.mean_fit = []
        self.best_values = []
        self.num_variables = len(upper_bound)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.total_bound = np.asarray(upper_bound) - np.asarray(lower_bound)
        self.domain = [upper_bound, lower_bound]
        self.gen = None
        self.sigma = None
        self.best_individual = None
        self.min_prob = min_prob  # min for p of mut and cross
        self.min_mut = min_mut

    def __scale_fitness_1(self, x):
        return x + np.abs(np.min(x))

    def __scale_fitness_2(self, x):
        return 1 / (1 + x)

    def __roulette_wheel_selection(self, cumulative_sum, n):
        ind = []
        r = np.random.uniform(0, 1, n)
        for i in range(0, n):
            index = 0
            while cumulative_sum[index] < r[i]:
                index += 1
            ind.append(index)
        return ind

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

    def __random_selection(self, n):
        return np.random.choice(range(0, n), n, replace=True)

    def __tournament_selection(self, fit, tourn_size, n):
        selection = []
        for i in range(0, n):
            competitors = np.random.choice(range(0, n), tourn_size, replace=False)
            best_index = np.argmax(fit[competitors])
            selection.append(competitors[best_index])
        return selection

    def __crossover_method_1(self, par):
        child = np.copy(par[0])
        for j in range(0, np.shape(par[0])[0]):
            child[j] = np.mean(par[:,j])
        return child

    def __crossover_method_2(self, par):
        child = np.copy(par[0])
        n = np.shape(par[0])[0]
        random_nums = np.random.randint(low=0, high=len(par), size=n)
        for j in range(0, n):
            child[j] = par[random_nums[j]][j]
        return child

    def __mutation_method_1(self, child, const_mutate):
        for j in range(0, np.shape(child)[0]):
            random_nums = np.random.uniform(-const_mutate[j], const_mutate[j], 1)
            child[j] = child[j] + random_nums[0]

    def __boltzmann_selection(self, f1, f2, t, find_max):
        p = 1 / (1+np.exp((f1-f2)/t))
        r = np.random.uniform(0, 1, 1)
        if find_max:
            if r <= p:
                return 1
            else:
                return 0
        else:
            if r <= p:
                return 0
            else:
                return 1

    def __reproduction1(self, par, f, p_cross, p_mutate, const_mutate, rep_method, par_battle, t, find_max):  # --- new arg ---
        p = np.random.uniform(0, 1, 2)
        if p[0] <= p_cross:
            if rep_method == 'average':
                child = self.__crossover_method_1(par)
            elif rep_method == 'intuitive':
                child = self.__crossover_method_2(par)
        else:
            if find_max:
                child = np.copy(par[np.argmax(f)])
            else:
                child = np.copy(par[np.argmin(f)])

        if p[1] <= p_mutate:
            self.__mutation_method_1(child, const_mutate)

        # --- new ---
        if par_battle:
            f3 = self.fitness_function(child)
            for i in range(0, len(f)):
                ind = self.__boltzmann_selection(f[i], f3, t, find_max)
                if ind == 1:
                    return child
            return np.copy(par[np.argmax(f)])
        return child

    def __reproduction2(self, par, sigma, f, p_cross, p_mutate, rep_method, par_battle, t, find_max):  # --- new arg ---
        p = np.random.uniform(0, 1, 2)
        if p[0] <= p_cross:
            if rep_method == 'average':
                c_values = self.__crossover_method_1(par)
                c_sigma = self.__crossover_method_1(sigma)
            elif rep_method == 'intuitive':
                c_values = self.__crossover_method_2(par)
                c_sigma = self.__crossover_method_2(sigma)
        else:
            if find_max:
                c_values = np.copy(par[np.argmax(f)])
                c_sigma = np.copy(sigma[np.argmax(f)])
            else:
                c_values = np.copy(par[np.argmin(f)])
                c_sigma = np.copy(sigma[np.argmin(f)])

        if p[1] <= p_mutate:
            c_values, c_sigma = self.__mutation_self_adaptive(c_values, c_sigma)

        # --- new ---
        if par_battle:
            f3 = self.fitness_function(c_values)
            for i in range(0, len(f)):
                ind = self.__boltzmann_selection(f[i], f3, t, find_max)
                if ind == 1:
                    return c_values, c_sigma
            return np.copy(par[np.argmax(f)]), np.copy(sigma[np.argmax(f)])
        return c_values, c_sigma

    def __logistic_decay(self, max_value, rate, index):
        p = (2 * max_value * max_value * np.exp(rate * index)) / (
                    max_value + max_value * np.exp(rate * index))
        return p

    def __bisection(self, max_iter, max_value, iter_index, min_value):
        a = -0.00001
        b = -0.5
        tol = 1e-7
        if (self.__logistic_decay(max_value, a, iter_index)-min_value) * (self.__logistic_decay(max_value, b, iter_index)-min_value) > 0:
            print("Error: Function requires a change of signs over given interval")
            return -1

        iter = 0
        while True:
            m = (a + b) / 2
            fa = self.__logistic_decay(max_value, a, iter_index)-min_value
            fm = self.__logistic_decay(max_value, m, iter_index)-min_value
            if fa * fm < 0:
                b = m
            elif fa * fm > 0:
                a = m
            else:  # fa*fm == 0 ; where fm == 0
                break

            iter += 1
            if (abs(b - a) < tol):
                break
            elif iter == max_iter:
                break
        return m

    def __find_rate(self, iter_index, max_value, min_value):
        return self.__bisection(iter_index=iter_index, max_value=max_value, min_value=min_value, max_iter=50)

    def evolve(self, p_cross, p_mutate, mutate_bound=0.1, sel_method='roulette', rep_method='average', p_method='static',
               mut_method='static', tourn_size=.10, elitism=0.0, max_iter=100, info=True, find_max=False,
               par_count=2, par_battle=False, warm_start=False):

        if not warm_start:
            self.mean_fit = []
            self.best_fit = []
            init_gen = np.empty(shape=(self.gen_size, self.num_variables))
            for i in range(0, self.num_variables):
                init_gen[:, i] = np.random.uniform(self.lower_bound[i], self.upper_bound[i], self.gen_size)
            self.gen = np.copy(init_gen)

            if mut_method == 'self-adaptive':
                init_sigma = np.empty(shape=(self.gen_size, self.num_variables))
                for i in range(0, self.num_variables):
                    init_sigma[:, i] = np.random.uniform(0.01 * self.total_bound[i],
                                                         0.2 * self.total_bound[i], self.gen_size)
                self.sigma = np.copy(init_sigma)

        bound_mut = mutate_bound * (np.abs(self.lower_bound) + np.abs(self.upper_bound))
        n, c = np.shape(self.gen)
        tourn_size = np.maximum(1, int(n * tourn_size))

        slope_c = (p_cross-self.min_prob)/(-max_iter)
        slope_m = (p_mutate-self.min_prob)/(-max_iter)
        slope_bound = (mutate_bound - self.min_mut) / (-max_iter)
        p_cross_line = np.arange(0, max_iter)*slope_c+p_cross
        p_mutate_line = np.arange(0, max_iter) * slope_m + p_mutate
        mutate_bound_line = np.arange(0, max_iter) * slope_bound + mutate_bound

        iter_index = int(0.90*max_iter)
        rate_mut = self.__find_rate(iter_index, p_mutate, self.min_prob)
        rate_cross = self.__find_rate(iter_index, p_cross, self.min_prob)
        rate_bound = self.__find_rate(iter_index, mutate_bound, self.min_mut)

        max_p_cross = p_cross  # new
        max_p_mutate = p_mutate  # new
        max_mutate_bound = mutate_bound  # new
        t = 100  # new
        if par_count != 2:
            par_battle = False

        for k in range(0, max_iter):
            fitness = self.fitness_function(self.gen)
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

            rel_fit = []
            for i in range(0, len(self.gen)):
                tourn = np.random.choice(range(0, n), tourn_size)
                rel_fit.append(self.__calculate_relative_fit(fitness[i], fitness[tourn], find_max))
            rel_fit = np.asarray(rel_fit)

            ind = range(0, n)
            if sel_method == 'roulette':
                fit_sum = np.sum(rel_fit)
                fit = rel_fit / fit_sum
            else:
                fit = rel_fit
            temp = np.column_stack((fit, ind))
            temp = temp[np.argsort(temp[:, 0])]
            if elitism != 0.0:
                best_index = int(n - np.ceil(n * elitism))
                best_ind = range(best_index, n)
                best_ind = np.array(temp[best_ind, 1], dtype=int).tolist()

            # ----- new -----
            selected = []
            for i in range(0, par_count):
                if sel_method == 'roulette':
                    cumulative_sum = np.cumsum(temp[:, 0])
                    selection = self.__roulette_wheel_selection(cumulative_sum, n)
                elif sel_method == 'tournament':
                    selection = self.__tournament_selection(fit, tourn_size, n)
                elif sel_method == 'random':
                    selection = self.__random_selection(n)
                selected.append(selection)
            selected = np.asarray(selected).T

            # ----- new ----
            if p_method == 'logistic':
                if p_cross <= self.min_prob:
                    p_cross = self.min_prob
                else:
                    p_cross = self.__logistic_decay(max_p_cross, rate_cross, k)

                if p_mutate <= self.min_prob:
                    p_mutate = self.min_prob
                else:
                    p_mutate = self.__logistic_decay(max_p_mutate, rate_mut, k)
            elif p_method == 'static':
                pass
            elif p_method == 'linear':
                p_mutate = p_mutate_line[k]
                p_cross = p_cross_line[k]

            if mut_method == 'logistic':
                if mutate_bound <= self.min_mut:
                    mutate_bound = self.min_mut
                else:
                    mutate_bound = self.__logistic_decay(max_mutate_bound, rate_bound, k)
                bound_mut = mutate_bound * (np.abs(self.lower_bound) + np.abs(self.upper_bound))
            elif mut_method == 'static':
                pass
            elif mut_method == 'linear':
                mutate_bound = mutate_bound_line[k]
                bound_mut = mutate_bound * (np.abs(self.lower_bound) + np.abs(self.upper_bound))

                # ------ end new -----

            if mut_method == 'self-adaptive':
                children = []
                children_sigma = []
                for i in range(0, n):
                    # -- new ---
                    c_value, c_sigma = self.__reproduction2(self.gen[selected[i]], self.sigma[selected[i]],
                                                         fitness[selected[i]],p_cross, p_mutate,rep_method,
                                                         par_battle, t, find_max)  # --- new arg ---
                    children.append(c_value)
                    children_sigma.append(c_sigma)
            else:
                children = []
                for i in range(0, n):
                    # -- new ---
                    children.append(self.__reproduction1(self.gen[selected[i]], fitness[selected[i]],
                                                 p_cross, p_mutate, bound_mut, rep_method, par_battle,
                                                 t, find_max))  # --- new arg ---

            # --- new ---
            if par_battle:
                prev_t = t
                alpha = np.random.uniform(0.95, 1, 1)[0]
                t = prev_t*alpha

            gen_next = np.asarray(children)
            if mut_method == 'self-adaptive':
                sigma_next = np.asarray(children_sigma)

            for i in range(0, n):
                for j in range(0, c):
                    if gen_next[i][j] > self.upper_bound[j]:
                        gen_next[i][j] = self.upper_bound[j]
                        if mut_method == 'self-adaptive':
                            sigma_next[i][j] *= 0.9
                    elif gen_next[i][j] < self.lower_bound[j]:
                        gen_next[i][j] = self.lower_bound[j]
                        if mut_method == 'self-adaptive':
                            sigma_next[i][j] *= 0.9

            if elitism != 0.0:
                fit_next = self.fitness_function(gen_next)
                ind = range(0, n)
                temp = np.column_stack((fit_next, ind))
                if find_max:
                    temp = temp[np.argsort(-temp[:, 0])]
                else:
                    temp = temp[np.argsort(temp[:, 0])]

                ind_replace = np.array(temp[best_index:n, 1], dtype=int).tolist()
                gen_next[ind_replace] = self.gen[best_ind]

            self.gen = gen_next
            if mut_method == 'self-adaptive':
                self.sigma = sigma_next

        fitness = self.fitness_function(gen_next)
        if find_max:
            self.best_individual = self.gen[np.argmax(fitness)]
        else:
            self.best_individual = self.gen[np.argmin(fitness)]

    def plot(self, starting_gen=0):
        x_range = range(starting_gen, len(self.best_fit))
        plt.plot(x_range, self.mean_fit[starting_gen:], label="Mean Fitness")
        plt.plot(x_range, self.best_fit[starting_gen:], label="Best Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Value")
        plt.suptitle("Mean and Best Fitness for Algorithm: ")
        plt.legend()
        plt.show()



