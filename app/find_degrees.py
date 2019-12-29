import numpy as np
from itertools import product
from multiprocessing import Pool
from copy import deepcopy


def brute(args):
    obj, i, j, k = args
    obj.deg = [i + 1, j + 1, k + 1]
    obj.built_A()
    obj.lamb()
    obj.psi()
    obj.built_a()
    obj.built_Fi()
    obj.built_c()
    obj.built_F()
    obj.built_F_()
    print(i,j,k)
    return (i, j, k), np.linalg.norm(obj.norm_error, np.inf), obj.norm_error

def determine_deg(a, p1, p2, p3):
    d = list(map(brute, product([a], p1, p2, p3)))
    best = d[0]
    for i in d:
        if i[1] < best[1]:
            best = i
    return best
