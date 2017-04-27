# -*- coding: utf-8 -*-
import time

from sklearn_MLP import grid_search
from sklearn_MLP import main as Bayesian_Optimization


def main():
    start_time1 = time.time()
    Bayesian_Optimization(0, "ucb") # kernel ... Matern(nu=0.5)
                                    # acquisition function ... ucb
    duration1 = time.time() - start_time1

    start_time2 = time.time()
    grid_search(verbose=False)
    duration2 = time.time() - start_time2

    print("Bayesian Optimization: \t %.3f sec" % duration1)
    print("Grid Search: \t %.3f sec" % duration2)

if __name__ == "__main__":
    main()
