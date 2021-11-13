# coding=utf-8
import copy
import random
import warnings

import numpy as np
from pyDOE2 import lhs

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from scipy.stats import norm

from sklearn.preprocessing import power_transform

# Need to import the searcher abstract class, the following are essential
from thpo.abstract_searcher import AbstractSearcher

from multiprocessing import Process
import multiprocessing


# configurations
De_duplication = False
Alpha = 1e-8
EnlargeRate = 1
ImproveStep = 6
Lambda = 1


class UtilityFunction(object):
    """
    This class mainly implements the collection function
    """

    def __init__(self, kind, kappa, x_i):
        self.kappa = kappa
        self.x_i = x_i

        self._iters_counter = 0

        if kind not in ['gp_ucb', 'gp_ei', 'gp_poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x_x, model, y_max):
        if self.kind == 'gp_ucb':
            return self._ucb(x_x, model, self.kappa)
        if self.kind == 'gp_ei':
            return self._ei(x_x, model, y_max, self.x_i)
        if self.kind == 'gp_poi':
            return self._poi(x_x, model, y_max, self.x_i)

    @staticmethod
    def _ucb(x_x, model, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = model.predict(x_x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x_x, g_p, y_max, x_i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            means = []
            stds = []
            for i in range(len(g_p)):
                mean, std = g_p[i].predict(x_x, return_std=True)
                means.append(mean)
                stds.append(std)
        maximum = None
        for i in range(len(means)):
            a_a = (means[i] - y_max - x_i)
            z_z = a_a / stds[i]
            x = a_a * norm.cdf(z_z) + std * norm.pdf(z_z)
            if maximum is None:
                maximum = x
            else:
                maximum[maximum < x] = x[maximum < x]
        return maximum

    @staticmethod
    def _poi(x_x, g_p, y_max, x_i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = g_p.predict(x_x, return_std=True)

        z_z = (mean - y_max - x_i) / std
        return norm.cdf(z_z)


class Searcher(AbstractSearcher):

    def __init__(self, parameters_config, n_iter, n_suggestion):
        """ Init searcher
        Args:
            parameters_config: parameters configuration, consistent with the definition of parameters_config of EvaluateFunction. dict type:
                    dict key: parameters name, string type
                    dict value: parameters configuration, dict type:
                        "parameter_name": parameter name
                        "parameter_type": parameter type, 1 for double type, and only double type is valid
                        "double_max_value": max value of this parameter
                        "double_min_value": min value of this parameter
                        "double_step": step size
                        "coords": list type, all valid values of this parameter.
                            If the parameter value is not in coords,
                            the closest valid value will be used by the judge program.
                    parameter configuration example, eg:
                    {
                        "p1": {
                            "parameter_name": "p1",
                            "parameter_type": 1
                            "double_max_value": 2.5,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0, 2.5]
                        },
                        "p2": {
                            "parameter_name": "p2",
                            "parameter_type": 1,
                            "double_max_value": 2.0,
                            "double_min_value": 0.0,
                            "double_step": 1.0,
                            "coords": [0.0, 1.0, 2.0]
                        }
                    }
                    In this example, "2.5" is the upper bound of parameter "p1", and it's also a valid value.
        n_iteration: number of iterations
        n_suggestion: number of suggestions to return
        """
        AbstractSearcher.__init__(self, parameters_config, n_iter, n_suggestion)

        self.gp_num = 1
        self.n_parameters = len(parameters_config)
        self.gp = []
        for i in range(self.gp_num):
            # d = np.random.normal(1, 1, self.n_parameters)
            # d[d < 0.2] = 0.2
            # d[d > 5] = 5
            d = 1
            Kernel = Matern(length_scale=d, nu=2.5)
            # print("length_scale =", Kernel.theta)
            # Kernel = Matern(nu=2.5)
            self.gp.append(GaussianProcessRegressor(
                kernel=Kernel,
                # kernel=Matern(length_scale=1.2, nu=0.5),
                alpha=Alpha,
                normalize_y=False,
                n_restarts_optimizer=5,
                random_state=np.random.RandomState(1),
            ))
        self.iter = 0
        self.improve_step = ImproveStep
        self.decay_rate = 0.99
        self.de_duplication = De_duplication


    # def init_param_group(self, n_suggestions):
    #     """ Suggest n_suggestions parameters in random form
    #     Args:
    #         n_suggestions: number of parameters to suggest in every iteration
    #     Return:
    #         next_suggestions: n_suggestions Parameters in random form
    #     """
    #     next_suggestions = [{p_name: p_conf['coords'][random.randint(0, len(p_conf["coords"]) - 1)]
    #                         for p_name, p_conf in self.parameters_config.items()} for _ in range(n_suggestions)]

    #     return next_suggestions

    def init_sample_param_group(self):
        nums_sample = len(list(self.parameters_config.keys()))
        if self.iter == 1:
            self.get_sample_points = lhs(nums_sample, 50, criterion="lhsmu", random_state=2021)

        zeros_array = np.zeros([self.n_suggestion, nums_sample])
        for _ in range(5 * (self.iter - 1), 5 * self.iter):
            get_sample_point = self.get_sample_points[_]
        
            for ind, k in enumerate(self.parameters_config.keys()):
                k_and_v = self.parameters_config[k]['coords']
                zeros_array[_ % self.n_suggestion, ind] = k_and_v[int((len(k_and_v) - 1) * get_sample_point[ind])]
        zeros_array = np.array(zeros_array)
        return zeros_array

    def parse_suggestions_history(self, suggestions_history):
        """ Parse historical suggestions to the form of (x_datas, y_datas), to obtain GP training data
        Args:
            suggestions_history: suggestions history
        Return:
            x_datas: Parameters
            y_datas: Reward of Parameters
        """
        suggestions_history_use = copy.deepcopy(suggestions_history)
        x_datas = [[item[1] for item in sorted(suggestion[0].items(), key=lambda x: x[0])]
                   for suggestion in suggestions_history_use]
        y_datas = [suggestion[1] for suggestion in suggestions_history_use]
        x_datas = np.array(x_datas)
        y_datas = np.array(y_datas)
        return x_datas, y_datas

    def train_gp(self, model, i, x_datas, y_datas):
        model.fit(x_datas, y_datas)
        return model

    def train_gps(self, x_datas, y_datas):
        """ train gp
        Args:
            x_datas: Parameters
            y_datas: Reward of Parameters
        Return:
            gp: Gaussian process regression
        """
        pool = multiprocessing.Pool(8)
        res_l = []
        for i in range(self.gp_num):
            p = pool.apply_async(self.train_gp, args=(self.gp[i], i, x_datas, y_datas))
            res_l.append(p)
        pool.close()
        pool.join()
        res = []
        for r in res_l:
            res.append(r.get())
        self.gp = res

    def random_sample(self):
        """ Generate a random sample in the form of [value_0, value_1,... ]
        Return:
            sample: a random sample in the form of [value_0, value_1,... ]
        """
        sample = [p_conf['coords'][random.randint(0, len(p_conf["coords"]) - 1)] for p_name, p_conf
                  in sorted(self.parameters_config.items(), key=lambda x: x[0])]
        return sample

    def get_bounds(self):
        """ Get sorted parameter space
        Return:
            _bounds: The sorted parameter space
        """
        def _get_param_value(param):
            value = [param['double_min_value'], param['double_max_value']]
            return value

        _bounds = np.array(
            [_get_param_value(item[1]) for item in sorted(self.parameters_config.items(), key=lambda x: x[0])],
            dtype=np.float
        )
        return _bounds

    def acq_max(self, f_acq, model, y_max, bounds, num_warmup, num_starting_points):
        """ Produces the best suggested parameters
        Args:
            f_acq: Acquisition function
            gp: GaussianProcessRegressor
            y_max: Best reward in suggestions history
            bounds: The parameter boundary of the acquisition function
            num_warmup: The number of samples randomly generated for the collection function
            num_starting_points: The number of random samples generated for scipy.minimize
        Return:
            Return the current optimal parameters
        """
        # Warm up with random points
        x_tries = np.array([self.random_sample() for _ in range(int(num_warmup))])
        # power_transform
        # x_tries = power_transform(x_tries, method='yeo-johnson')
        ys = f_acq(x_tries, model=model, y_max=y_max)

        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()
        # Explore the parameter space more throughly
        x_seeds = np.array([self.random_sample() for _ in range(int(num_starting_points))])
        # power_transform
        # x_seeds = power_transform(x_seeds, method='yeo-johnson')
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res = minimize(lambda x: -f_acq(x.reshape(1, -1), model=model, y_max=y_max),
                           x_try.reshape(1, -1),
                           bounds=bounds,
                           method="L-BFGS-B",
                          #  tol=1e-6
                           )
            # See if success
            if not res.success:
                continue
            # Store it if better than previous minimum(maximum).
            if max_acq is None or -res.fun[0] >= max_acq:
                x_max = res.x
                max_acq = -res.fun[0]
        return np.clip(x_max, bounds[:, 0], bounds[:, 1]), max_acq

    def parse_suggestions(self, suggestions):
        """ Parse the parameters result
        Args:
            suggestions: Parameters
        Return:
            suggestions: The parsed parameters
        """
        def get_param_value(p_name, value):
            p_coords = self.parameters_config[p_name]['coords']
            if value in p_coords:
                return value
            else:
                subtract = np.abs([p_coord - value for p_coord in p_coords])
                min_index = np.argmin(subtract, axis=0)
                return p_coords[min_index]

        p_names = [p_name for p_name, p_conf in sorted(self.parameters_config.items(), key=lambda x: x[0])]
        suggestions = [{p_names[index]:suggestion[index] for index in range(len(suggestion))}
                       for suggestion in suggestions]

        suggestions = [{p_name: get_param_value(p_name, value) for p_name, value in suggestion.items()}
                       for suggestion in suggestions]
        return suggestions

    def get_valid_suggestion(self, suggestion):

        def get_param_value(p_name, value):
            p_coords = self.parameters_config[p_name]['coords']
            if value in p_coords:
                return value
            else:
                subtract = np.abs([p_coord - value for p_coord in p_coords])
                min_index = np.argmin(subtract, axis=0)
                return p_coords[min_index]

        p_names = [p_name for p_name, p_conf in sorted(self.parameters_config.items(), key=lambda x: x[0])]
        suggestion = {p_names[index]: suggestion[index] for index in range(len(suggestion))}
        suggestion = [get_param_value(p_name, value) for p_name, value in suggestion.items()]
        return suggestion

    def get_single_suggest(self, index, y_datas, _bounds):
        # x_i = index * self.improve_step + np.random.rand() * self.improve_step
        x_i = index * self.improve_step
        utility_function = UtilityFunction(kind="gp_ei", kappa=(index + 1) * 2.567, x_i=x_i)
        suggestion, max_acq = self.acq_max(
            f_acq=utility_function.utility,
            model=self.gp,
            y_max=y_datas.max(),
            bounds=_bounds,
            num_warmup=1000,
            num_starting_points=5
        )
        return suggestion, max_acq

    def suggest_old(self, suggestions_history, n_suggestions=1):
        """ Suggest next n_suggestion parameters.
        Args:
            suggestions_history: a list of historical suggestion parameters and rewards, in the form of
                    [[Parameter, Reward], [Parameter, Reward] ... ]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}
                        Reward: a float type value
                    The parameters and rewards of each iteration are placed in suggestions_history in the order of iteration.
                        len(suggestions_history) = n_suggestion * iteration(current number of iteration)
                    For example:
                        when iteration = 2, n_suggestion = 2, then
                        [[{'p1': 0, 'p2': 0, 'p3': 0}, -222.90621774147272],
                         [{'p1': 0, 'p2': 1, 'p3': 3}, -65.26678723205647],
                         [{'p1': 2, 'p2': 2, 'p3': 2}, 0.0],
                         [{'p1': 0, 'p2': 0, 'p3': 4}, -105.8151893979122]]
            n_suggestion: int, number of suggestions to return
        Returns:
            next_suggestions: list of Parameter, in the form of
                    [Parameter, Parameter, Parameter ...]
                        Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                            {'p1': 0, 'p2': 0, 'p3': 0}
                    For example:
                        when n_suggestion = 3, then
                        [{'p1': 0, 'p2': 0, 'p3': 0},
                         {'p1': 0, 'p2': 1, 'p3': 3},
                         {'p1': 2, 'p2': 2, 'p3': 2}]
        """
        self.iter += 1
        if self.iter in [1,2]:
          suggestions = self.init_sample_param_group()
          next_suggestions = self.parse_suggestions(suggestions)
          return next_suggestions

        # if (suggestions_history is None) or (len(suggestions_history) <= 0):
        #     next_suggestions = self.init_param_group(n_suggestions)

        else:
            x_datas, y_datas = self.parse_suggestions_history(suggestions_history)
            for i in range(self.gp_num):
                # d = np.random.normal(1, 1, self.n_parameters)
                # d[d < 0.2] = 0.2
                # d[d > 5] = 5
                d = 1
                Kernel = Matern(length_scale=d, nu=2.5)
                # print("length_scale =", Kernel.theta)
                # Kernel = Matern(nu=2.5)
                self.gp[i].kernel = Kernel
            # print("x_datas=" ,x_datas)
            # print("y_datas=", y_datas)
            self.train_gps(x_datas, y_datas)
            _bounds = self.get_bounds()
            suggestions = []
            suggestions_candidate = []
            y_max = y_datas.max()
            pool = multiprocessing.Pool(10)
            res_l = []
            for index in range(n_suggestions+2):
                p = pool.apply_async(self.get_single_suggest, args=(index, y_datas, _bounds))
                res_l.append(p)
            pool.close()
            pool.join()
            for index in range(len(res_l)):
                r = res_l[index]
                suggestion, max_acq = r.get()
                suggestions_candidate.append((suggestion, max_acq * (Lambda * index + 1)))

            suggestions_candidate.sort(key=lambda y: y[1], reverse=True)
            if self.de_duplication:
                i = 0
                j = 0
                while i < n_suggestions and j < len(suggestions_candidate):
                    t = False
                    for data in x_datas:
                        if (data == np.array(self.get_valid_suggestion(suggestions_candidate[j][0]))).all():
                            t = True
                            break
                    if t:
                        j += 1
                    else:

                        suggestions.append(suggestions_candidate[j][0])
                        i += 1
                        j += 1
                while i < n_suggestions:
                    suggestions.append(self.random_sample())
                    i += 1
            else:
              for i in range(n_suggestions):
                suggestions.append(suggestions_candidate[i][0])
            suggestions = np.array(suggestions)
            suggestions = self.parse_suggestions(suggestions)
            next_suggestions = suggestions
            print("y_max =", y_max)
            print("improve_step =", self.improve_step)

        return next_suggestions

    # 不使用早停的suggest
    def suggest(self, day, running_suggestions, suggestion_history, n_suggestions=1):
      new_suggestions_history = []
      # for _ in range(n_suggestions):
      #   self.improve_step *= self.decay_rate
      for suggestion in suggestion_history:
        new_suggestions_history.append([suggestion["parameter"], suggestion['reward'][-1]['value']])
      return self.suggest_old(new_suggestions_history, n_suggestions)

    # #使用早停的suggest
    # @staticmethod
    # def get_my_score(reward):
    #     """ Get the most trusted reward of all iterations.
    #     Returns:
    #         most_trusted_reward: float
    #     """
    #     return reward[-1]['value']

    # @staticmethod
    # def get_my_score_lower_bound(reward):
    #     """ Get the most trusted reward of all iterations.
    #     Returns:
    #         lower_bound_reward: float
    #     """
    #     return reward[-1]['lower_bound']

    # @staticmethod
    # def get_my_score_upper_bound(reward):
    #     return reward[-1]['upper_bound']

    # def suggest(self, iteration_number, running_suggestions, suggestion_history, n_suggestions=1):
    #     """ Suggest next n_suggestion parameters. new implementation of final competition
    #     Args:
    #         iteration_number: int ,the iteration number of experiment, range in [1, 140]
    #         running_suggestions: a list of historical suggestion parameters and rewards, in the form of
    #                 [{"parameter": Parameter, "reward": Reward}, {"parameter": Parameter, "reward": Reward} ... ]
    #             Parameter: a dict in the form of {name:value, name:value, ...}. for example:
    #                 {'p1': 0, 'p2': 0, 'p3': 0}
    #             Reward: a list of dict, each dict of the list corresponds to an iteration,
    #                 the dict is in the form of {'value':value,  'upper_bound':upper_bound, 'lower_bound':lower_bound} 
    #                 Reward example:
    #                     [{'value':1, 'upper_bound':2,   'lower_bound':0},   # iter 1
    #                      {'value':1, 'upper_bound':1.5, 'lower_bound':0.5}  # iter 2
    #                     ]
    #         suggestion_history: a list of historical suggestion parameters and rewards, in the same form of running_suggestions
    #         n_suggestions: int, number of suggestions to return
    #     Returns:
    #         next_suggestions: list of Parameter, in the form of
    #                 [Parameter, Parameter, Parameter ...]
    #                     Parameter: a dict in the form of {name:value, name:value, ...}. for example:
    #                         {'p1': 0, 'p2': 0, 'p3': 0}
    #                 For example:
    #                     when n_suggestion = 3, then
    #                     [{'p1': 0, 'p2': 0, 'p3': 0},
    #                      {'p1': 0, 'p2': 1, 'p3': 3},
    #                      {'p1': 2, 'p2': 2, 'p3': 2}]
    #     """
        # MIN_TRUSTED_ITERATION = 8
        # new_suggestions_history = []
        # # for _ in range(n_suggestions):
        # #     self.improve_step *= self.decay_rate
        # for suggestion in suggestion_history:
        #     iterations_of_suggestion = len(suggestion['reward'])
        #     if iterations_of_suggestion >= MIN_TRUSTED_ITERATION:
        #         cur_score = self.get_my_score(suggestion['reward'])
        #         new_suggestions_history.append([suggestion["parameter"], cur_score])
        # return self.suggest_old(new_suggestions_history, n_suggestions)

    def is_early_stop(self, iteration_number, running_suggestions, suggestion_history):
        """ Decide whether to stop the running suggested parameter experiment.
        Args:
            iteration_number: int, the iteration number of experiment, range in [1, 140]
            running_suggestions: a list of historical suggestion parameters and rewards, in the form of
                    [{"parameter": Parameter, "reward": Reward}, {"parameter": Parameter, "reward": Reward} ... ]
                Parameter: a dict in the form of {name:value, name:value, ...}. for example:
                    {'p1': 0, 'p2': 0, 'p3': 0}
                Reward: a list of dict, each dict of the list corresponds to an iteration,
                    the dict is in the form of {'value':value,  'upper_bound':upper_bound, 'lower_bound':lower_bound} 
                    Reward example:
                        [{'value':1, 'upper_bound':2,   'lower_bound':0},   # iter 1
                         {'value':1, 'upper_bound':1.5, 'lower_bound':0.5}  # iter 2
                        ]
            suggestion_history: a list of historical suggestion parameters and rewards, in the same form of running_suggestions
        Returns:
            stop_list: list of bool, indicate whether to stop the running suggestions.
                    len(stop_list) must be the same as len(running_suggestions), for example:
                        len(running_suggestions) = 3, stop_list could be : 
                            [True, True, True] , which means to stop all the three running suggestions
        """

        # Early Stop algorithm demo 2:
        #
        #   If there are 3 or more suggestions which had more than 7 iterations,
        #   the worst running suggestions will be stopped
        #
        # MIN_ITERS_TO_STOP = 8
        # MIN_SUGGUEST_COUNT_TO_STOP = 5
        # MAX_ITERS_OF_DATASET = self.n_iteration
        # ITERS_TO_GET_STABLE_RESULT = 14
        # INITIAL_INDEX = -1

        # res = [False] * len(running_suggestions)
        # if iteration_number + ITERS_TO_GET_STABLE_RESULT <= MAX_ITERS_OF_DATASET:
        #     score_min_idx = INITIAL_INDEX
        #     score_min = float("inf")
        #     count = 0
        #     # Get the worst suggestion of current running suggestions
        #     for idx, suggestion in enumerate(running_suggestions):
        #         if len(suggestion['reward']) >= MIN_ITERS_TO_STOP:
        #             count = count + 1
        #             cur_score = self.get_my_score_upper_bound(suggestion['reward'])
        #             if score_min_idx == INITIAL_INDEX or cur_score <= score_min:
        #                 score_min_idx = idx
        #                 score_min = cur_score
        #     # Stop the worst suggestion
        #     if count >= MIN_SUGGUEST_COUNT_TO_STOP and score_min_idx != INITIAL_INDEX:
        #         res[score_min_idx] = True
        # return res
        # 不使用早停
        res = [False] * len(running_suggestions)
        return res