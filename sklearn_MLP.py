# -*- coding: utf-8 -*-

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from b.bayesian_optimization import BayesianOptimization

import sys
from collections import OrderedDict

#data
digits = datasets.load_digits()
n_sample = len(digits.images)
X = digits.images.reshape((n_sample, -1))
y = digits.target

def MLP(alpha, lr, layer1, layer2, layer3, layer4, layer5):
    # alpha ... L2 penalty
    # lr ... learning_rate
    mlp = MLPClassifier(hidden_layer_sizes=(int(layer1), int(layer2), int(layer3), int(layer4), int(layer5)),
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
    BO = BayesianOptimization(MLP,
                              {"alpha": (1e-8, 1e-4), "lr": (1e-6, 1e-2),
                               "layer1": (10, 100),"layer2": (10, 100),"layer3": (10, 100),"layer4": (10, 100),"layer5": (10, 100)},
                              verbose=1, kernel_num = k_num)

    BO.explore({"alpha": [1e-8, 1e-8, 1e-4, 1e-4],"lr": [1e-6, 1e-2, 1e-6, 1e-2],
                "layer1": [10, 50, 100, 50], "layer2": [10, 50, 100, 50],"layer3": [10, 50, 100, 50],"layer4": [10, 50, 100, 50],"layer5": [10, 50, 100, 50] })

    BO.maximize(n_iter=10, acq=acq, **gp_params)
    sys.exit()

    print("-"*53)
    print("Final Results")
    print("kernel: {}".format(str(BO.kernel)))
    print("score: {}".format(BO.res["max"]["max_val"]))
    print("best_parameter: ")
    print(BO.res["max"]["max_params"])
    print("acquisition function")
    print(BO.acquisition)

    final_result = OrderedDict( (("kernel", str(BO.kernel)),
                                 ("acquisition", BO.acquisition),
                                 ("value", BO.res['max']['max_val'])) )

    final_result.update(BO.res["max"]["max_params"])
    return final_result



if __name__ == "__main__":
    main(0, "ucb")
