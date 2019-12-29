import numpy as np

def conjugate_gradient_method(A, b, eps):
    '''
    Conjugate Gradient Method that solve equation Ax = b with given accuracy
    :param A:matrix A
    :param b:vector b
    :param eps: accuracy
    :return: solution x
    '''
    return gradient_descent(A, b, eps)
    # n = len(A.T) # number column
    # xi1 = xi = np.zeros(shape=(n,1), dtype=float)
    # vi = ri = b # start condition
    # i = 0 #loop for number iteration
    # N = 10000 #maximum of iteration
    # while True:
    #     try:
    #         i+= 1
    #         ai = float(vi.T*ri)/float(vi.T*A*vi) # alpha i
    #         xi1 = xi+ai*vi # x i+1
    #         ri1 = ri-ai*A*vi # r i+1
    #         betai = -float(vi.T*A*ri1)/float(vi.T*A*vi) # beta i
    #         vi1 = ri1+betai*vi
    #         xi,vi,ri = xi1,vi1,ri1
    #         if (np.linalg.norm(ri1,np.inf)<eps):
    #             break
    #         if i==N:
    #             raise NameError('Over index: many iterations')
    #     except NameError:
    #         print("conjugate_gradient_method is in 1000 iteration")
    # return np.matrix(xi1)

def conjugate_gradient_method_v2(A, b, eps):
    '''
    Conjugate Gradient Method that solve equation Ax = b with given accuracy
    :param A:matrix A
    :param b:vector b
    :param eps: accuracy
    :return: solution x
    '''
    return gradient_descent(A, b, eps)
    # n = len(A.T) # number column
    # xi = np.zeros(shape=(n,1), dtype=float)
    # xi1 = xi.copy()
    # x_best = xi.copy()
    # vi = ri = b # start condition
    # resid_best_norm = np.linalg.norm(ri, np.inf)
    # i = 0 #loop for number iteration
    # while True:
    #     i+= 1
    #     ai = float(vi.T*ri)/float(vi.T*A*vi) # alpha i
    #     xi1 = xi+ai*vi # x i+1
    #     ri1 = ri-ai*A*vi # r i+1
    #     betai = -float(vi.T*A*ri1)/float(vi.T*A*vi) # beta i
    #     vi1 = ri1+betai*vi
    #     xi,vi,ri = xi1,vi1,ri1
    #     resid_current_norm = np.linalg.norm(ri,np.inf)
    #     if resid_current_norm < resid_best_norm:
    #         resid_best_norm = resid_current_norm
    #         x_best = xi
    #     if (resid_best_norm<eps) or i > 10 * n:
    #         break
    # return np.matrix(x_best)

def conjugate_gradient_method_v3(A, b, eps):
    '''
    Conjugate Gradient Method that solve equation Ax = b with given accuracy
    :param A:matrix A
    :param b:vector b
    :param eps: accuracy
    :return: solution x
    '''
    return gradient_descent(A, b, eps)
    # x = np.zeros((A.shape[0],1))
    # p = rnext = rcur = b - A * x
    # while np.linalg.norm(rcur) > eps:
    #     rcur = rnext
    #     alpha = np.linalg.norm(rcur)**2 / float(p.T * A * p)
    #     x = x + alpha * p
    #     rnext = rcur - alpha * (A * p)
    #     if np.linalg.norm(rnext) > eps:
    #         beta = np.linalg.norm(rnext)**2 / np.linalg.norm(rcur)**2
    #         p = rnext + beta * p
    # return np.matrix(x)


def gradient_descent(A, b, eps):
    m = len(A.T)
    x = np.zeros(shape=(m,1))
    i = 0
    imax = 1000
    r = b - A * x
    delta = r.T * r
    delta0 = delta
    while i < imax and delta > eps ** 2 * delta0:
        alpha = float(delta / (r.T * (A * r)))
        x = x + alpha * r
        r = b - A * x
        delta = r.T * r
        i += 1

    return x
