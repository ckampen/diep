import numpy as np

W = np.matrix([[1,0,0],[-2,1,0],[0,0,1]])
X = np.matrix([[2,4,-2], [4,9,-3], [-2, -3,7]])

Y = W * X
print(Y)

#Naive implementation of matnul
def matmul(a, b):
    x = []
    for i, ai in enumerate(a):
        print("AI")
        print(ai)
        j = 0
        xi = []

        n = len(ai) -1
        m = len(b) -1
        bij = 0

        aii = 0
        bii = 0

        while aii < n:
            aij = ai[aii]
            aii = aii + 1



        for bi in b:
            v = 0
            for aij, _  in enumerate(ai):
                print("aij: %d" % aij)
                print("bij: %d" % bij)
                print("ai")
                print(ai)
                print("bi")
                print(bi)
                print("------")
                print("[bij]: %d" % bi[bij])
                print("[aij]: %d" % ai[aij])
                v = v + ai[aij] * bi[bij]
                print(v)
                print("------")
            xi.append(v)
        bij = bij + 1
        j = j + 1
        x.append(xi)
    return x

W = [[1,0,0],[1,0,0],[0,0,1]]
X = [[2,4,-2], [4,9,-3], [-2,-3,7]]
Y = matmul(W,X)
print(Y)
