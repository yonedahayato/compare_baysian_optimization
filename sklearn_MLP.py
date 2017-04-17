# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import b.bayesian_optimization import BayesianOptimization

import sys

#data
digits = datasets.load_digits()
n_sample = len(digits.images)
X = digits.images.reshape((n_sample, -1))
y = digits.target

def MLP(alpha):
    # alpha ... L2 penalty
    # lr ... learning_rate
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100),
                        max_iter=400, solver="sgd",
                        alpha=alpha, learning_rate_init=lr)

    train_size = int(n_sample*0.8)
    X_train, X_test = X[:train_size, :], X[train_size:, :]
    y_train, y_test = y[:train_size], y[train_size:]

    mlp.fit(X_train, y_train)
    #print("Test set score: {}".format(mlp.score(X_test, y_test)))
    return mlp.score(X_test, y_test)

def main(k_num, acq):
    gp_params = {"alpha": 1e-5}
    BO = BayesianOptimization(MPL,
                              {"alpha": (1e-8, 1e-4), "lr": (1e-6, 1e-2)},
                              verbose=1,kernel_num = k_num)
    BO.explore({"alpha": [1e-8, 1e-8, 1e-4, 1e-4],
                "lr": [1e-6, 1e-2, le-6, le-2]})

    BO.maximize(n_iter=10, acq=acq, **gp_params)

    print("-"*53)
    print("Final Results")
    print("kernel: {}".format(str(BO.kernel)))
    print("score: {}".format(BO.res["max"]["max_val"]))
    print("best_parameter: ")
    print(BO.res["max"]["max_params"])
    print("acquisition function")
    print(BO.acquisition)

    final_result.update(BO.res["max"]["max_params"])
    return final_result



if __name__ == "__main__":
    main()