# -*- coding: utf-8 -*-

from sklearn_example import main as compare_kernel

def main():
    result = []
    for i in range(4):
        result.append(compare_kernel(i))


    print("-"*100)
    category = []
    category_size=[35,15,15,15]

    for i, k in enumerate(result[0].keys()):
        category.append(k)
        print("{0:>{1}}".format(k,category_size[i]), end="|")
    print("")

    for r in result:
        for i, c in enumerate(category):
            if isinstance(r[c], str):
                print("{0:>{1}}".format(r[c], category_size[i]),end="|")
            else:
                print("{0:>{1}.{2}f}".format(r[c], category_size[i], 5,5),end="|")
        print("")


if __name__ == "__main__":
    main()
