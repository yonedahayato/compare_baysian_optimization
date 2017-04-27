# -*- coding: utf-8 -*-
import inspect
import sys

def grid_search(function, tuned_parameters, verbose=True):

    key_list = []
    max_index = []
    for k in tuned_parameters.keys():
        key_list.append(k)
        max_index.append(len(tuned_parameters[k]))
    #print(key_list)

    def execute(index, function, verbose=True):
        # argument と function を用意して実行する #
        # return...score（未実装）
        if verbose: print(index)

        arg_str = ""
        for i in range(len(key_list)):
            Index = index[i]
            arg_str += (key_list[i]+"="+str( tuned_parameters[key_list[i]][Index] ))
            if i != len(key_list)-1:
                arg_str += ","

        Score = eval("function("+arg_str+")")
        if verbose: print(Score)
        return Score

    def execute_for(index, j, Best_Score, Best_Index, verbose=False):
        for i in range(max_index[j]):
            if j==0:
                Score = execute(index, function, verbose=verbose)
                if Best_Score < Score:
                    Best_Score = Score
                    Best_Index = index
                index[j] += 1
            else:
                index, Best_Score, Best_Index = execute_for(index, j-1, Best_Score, Best_Index, verbose=verbose)

        index[j] = 0
        if j+1 == len(index): pass
        else: index[j+1] += 1

        return index, Best_Score, Best_Index

    Best_Score = 0
    Best_Index = None

    index = [0]*len(key_list) # start index
    index, Best_Score, Best_Index = execute_for(index, len(key_list)-1,
                                                Best_Score, Best_Index, verbose=verbose)

    print("===== Grid Search =====")
    print("Best_Score: {}".format(Best_Score))
    Best_Parameters = {}
    for i, k in enumerate(key_list):
        Best_Parameters[k] = tuned_parameters[k][Best_Index[i]]
    print(Best_Parameters)
    print("===== ==== ====== =====")
