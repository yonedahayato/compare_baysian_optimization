# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from sklearn.datasets import make_classification, load_iris
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

#from bayes_opt import BayesianOptimization
import sys
from b.bayesian_optimization import BayesianOptimization

"""
# Load data set and target values
data, target = make_classification(
    n_samples=1000,
    n_features=45,
    n_informative=12,
    n_redundant=7
)
"""
# iris data set
iris = load_iris()
data = iris.data
target = iris.target



# 目的関数(SVC)
def svccv(C, gamma):
    val = cross_val_score(
        SVC(C=C, gamma=gamma, random_state=2),
        data, target, 'accuracy', cv=5
    ).mean()

    return val

# 目的関数(RFC)
def rfccv(n_estimators, min_samples_split, max_features):
    val = cross_val_score(
        RFC(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999),
            random_state=2
        ),
        data, target, 'f1', cv=2
    ).mean()
    return val

def main(k_num, acq):
    gp_params = {"alpha": 1e-5}
    svcBO = BayesianOptimization(svccv,
                                 {'C': (0.001, 100), 'gamma': (0.0001, 0.1)},
                                 verbose=1,
                                 kernel_num=k_num)
    svcBO.explore({'C': [0.001, 0.01, 0.1], 'gamma': [0.001, 0.01, 0.1]})
    """
    rfcBO = BayesianOptimization(
        rfccv,
        {'n_estimators': (10, 250),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999)}
    )
    """

    svcBO.maximize(n_iter=10, acq=acq, **gp_params)
    #print('-' * 53)
    #rfcBO.maximize(n_iter=10, **gp_params)

    print('-' * 53)
    print('Final Results')
    print('kernel: {}'.format(str(svcBO.kernel)))
    print('SVC: %f' % svcBO.res['max']['max_val'])
    #print('RFC: %f' % rfcBO.res['max']['max_val'])
    print('best_parameter: ')
    print(svcBO.res['max']['max_params'])
    print('acquisition function')
    print(svcBO.acquisition)

    final_result = {"kernel": str(svcBO.kernel),
                    "acquisition": svcBO.acquisition,
                    "value": svcBO.res['max']['max_val']}

    final_result.update(svcBO.res['max']['max_params'])
    return final_result

if __name__ == "__main__":
    main(0)

