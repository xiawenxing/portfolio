# MV Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk
import numpy as np
import os
import cvxopt as opt
from cvxopt import blas,solvers,matrix
from data_load.stocks import Stocks
from trade.portfolio import Portfolio


span_t = 120


def MV_weight_compute(n, context):
    w = np.zeros(n)
    R = context["R"].T
    
    N = 100
    mus = [10**(5.0*t/N-1.0) for t in range(N)]

    S = opt.matrix(np.cov(R))
    pbar = opt.matrix(np.mean(R,axis=1))

    G = -opt.matrix(np.eye(n))
    h = opt.matrix(0.0, (n,1))
    A = opt.matrix(1.0, (1,n))
    b = opt.matrix(1.0)

    prtflio = [solvers.qp(mu*S,-pbar, G, h ,A, b)['x'] for mu in mus]#有效边界
    re = [blas.dot(pbar,x) for x in prtflio]#有效边界的收益
    risk = [np.sqrt(blas.dot(x, S*x)) for x in prtflio]#有效边界的收益
    m1 = np.polyfit(re, risk, 2)
    x1 = np.sqrt(m1[2]/m1[0])
    w = np.asarray(np.squeeze(solvers.qp(opt.matrix(x1*S), -pbar, G, h, A, b)['x']))
    
    return w


if __name__ == "__main__":
    print("this is MV Portfolio")
