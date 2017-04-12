# -*- coding: utf-8 -*-

from sklearn_example import main as compare_kernel

def main():
    result = []
    for i in range(4):
        result.append(compare_kernel(i))
        if i==2: break


    print("-"*50)
    category = []
    for k in result[0].keys():
        category.append(k)
        print("\t{}\t".format(k), end="|")
    print("")

    for r in result:
        for c in category:
            print(r[c], end="|")
        print("")



if __name__ == "__main__":
    main()
