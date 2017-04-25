# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import GridSearchCV
from b.bayesian_optimization import BayesianOptimization

import sys
from collections import OrderedDict
from sklearn.svm import SVC


#data
digits = datasets.load_digits()
n_sample = len(digits.images)
X = digits.images.reshape((n_sample, -1))
y = digits.target

train_size = int(n_sample*0.8)
X_train, X_test = X[:train_size, :], X[train_size:, :]
y_train, y_test = y[:train_size], y[train_size:]


def MLP(alpha, lr, layer1, layer2, layer3):
    # :::hyper parameters:::
    #
    # alpha ... L2 penalty
    # lr ... learning_rate
    # layer_n

    mlp = MLPClassifier(hidden_layer_sizes=(int(layer1), int(layer2), int(layer3)),
                        max_iter=400, solver="sgd",
                        alpha=alpha, learning_rate_init=lr)

    mlp.fit(X_train, y_train)
    #print("Test set score: {}".format(mlp.score(X_test, y_test)))
    return mlp.score(X_test, y_test)


def grid_search(verbose=True):
    tuned_parameters = {"alpha": [10**i for i in range(-8, -4+1, 1)],
                        "lr": [10**i for i in range(-6, -2+1, 1)],
                        "layer1": [10*i for i in range(1, 10+1, 1)],
                        "layer2": [10*i for i in range(1, 10+1, 1)],
                        "layer3": [10*i for i in range(1, 10+1, 1)]}

    from GridSearch.grid_search import grid_search
    grid_search(MLP, tuned_parameters, verbose=verbose)


def main(k_num, acq):
    gp_params = {"alpha": 1e-5}
    BO = BayesianOptimization(MLP,
                              {"alpha": (1e-8, 1e-4), "lr": (1e-6, 1e-2),
                               "layer1": (10, 100),"layer2": (10, 100),"layer3": (10, 100)},
                              verbose=1, kernel_num = k_num)

    BO.explore({"alpha": [1e-8, 1e-8, 1e-4, 1e-4],"lr": [1e-6, 1e-2, 1e-6, 1e-2],
                "layer1": [10, 50, 100, 50], "layer2": [10, 50, 100, 50],"layer3": [10, 50, 100, 50]})

    BO.maximize(n_iter=10, acq=acq, **gp_params)

    print("-"*53)
    print("Final Results")
    print("kernel: {}".format(str(BO.kernel)))
    print("acquisition function: {}".format(BO.acquisition))

    print("score: {}".format(BO.res["max"]["max_val"]))
    print("best_parameter: ")
    print(BO.res["max"]["max_params"])

    final_result = OrderedDict( (("kernel", str(BO.kernel)),
                                 ("acquisition", BO.acquisition),
                                 ("value", BO.res['max']['max_val'])) )

    final_result.update(BO.res["max"]["max_params"])
    return final_result


if __name__ == "__main__":
    #main(0, "ucb")  # Bayesian Optimization
                    # kernel...Matern(nu=0.5)
                    # acquisition function...ucb

    grid_search(verbose=True) # 比較用の Grid Search
