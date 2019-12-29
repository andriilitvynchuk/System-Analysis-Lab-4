class _Polynom(object):
    def __init__(self, ar, symbol = 'x',eps = 1e-15):
        self.ar = ar
        self.symbol = symbol
        self.eps = eps

    def __repr__(self):
        #joinder[first, negative] = str
        joiner = {
            (True, True):'-',
            (True, False): '',
            (False, True): ' - ',
            (False, False): ' + '
        }

        result = []
        for deg, coef in reversed(list(enumerate(self.ar))):
            sign = joiner[not result, coef < 0]
            coef  = abs(coef)
            if coef == 1 and deg != 0:
                coef = ''
            if coef < self.eps:
                continue
            f = {0: '{}{}', 1: '{}{}'+self.symbol}.get(deg, '{}{}'+ self.symbol +'^{}')
            result.append(f.format(sign, coef, deg))
        return ''.join(result) or '0'

#s = _Polynom([3,4,0,5,0,12], 'X').__repr__()
#print(s+s)

