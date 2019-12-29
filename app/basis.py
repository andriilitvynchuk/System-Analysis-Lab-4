from numpy.polynomial import Polynomial as pm


def basis_sh_chebyshev(degree):
    basis = [pm([-1, 2]), pm([1])]
    for i in range(degree):
        basis.append(pm([-2, 4])*basis[-1] - basis[-2])
    del basis[0]
    return basis


def basis_sh_legendre(degree):
    basis = [pm([1])]
    for i in range(degree):
        if i == 0:
            basis.append(pm([-1, 2]))
            continue
        basis.append((pm([-2*i - 1, 4*i + 2])*basis[-1] - i * basis[-2]) / (i + 1))
    return basis


def basis_hermite(degree):
    basis = [pm([0]), pm([1])]
    for i in range(degree):
        basis.append(pm([0,2])*basis[-1] - 2 * i * basis[-2])
    del basis[0]
    return basis


def basis_laguerre(degree):
    basis = [pm([1])]
    for i in range(degree):
        if i == 0:
            basis.append(pm([1, -1]))
            continue
        basis.append(pm([2*i + 1, -1])*basis[-1] - i * i * basis[-2])
    return basis

def basis_chebyshev(degree):
    basis = [pm([1])]
    for i in range(degree):
        if i == 0:
            basis.append(pm([0,1]))
            continue
        basis.append(pm([0,2]) * basis[-1] - basis[-2])
    return basis

def basis_sh_chebyshev_2(degree):
    basis = [pm([1])]
    for i in range(degree):
        if i == 0:
            basis.append(pm([-2,4]))
            continue
        basis.append(pm([-2, 4]) * basis[-1] - basis[-2])
    return basis

def basis_sh_chebyshev_2_shrinked(degree):
    basis = basis_sh_chebyshev_2(degree)
    for i in range(degree):
        basis[i] /= (i + 1)
    return basis