# UCB Portfolio
# Inputs: m, n,  τ（span_t）

# Output:
# The portfolio weight vectors ωk
import numpy as np
import os
import random
from data_load.stocks import Stocks
from trade.portfolio import Portfolio


span_t = 120


def UCB_weight_compute(n, context):
    
    R = context["R"].T
    w = np.zeros(n)
    ucb = np.zeros(n)
    select = np.zeros(n)
    rewards = np.asarray(np.squeeze(R.T[0,:]))

    N = len(R.T)
    for i in range(N):
        for j in range(n):
            if(select[j]==0):	
                ucb[j] = rewards[j]+1e400
            else:				
                ucb[j] = rewards[j]/select[j]+np.sqrt(3/2*np.log(i+1)/select[j])
        index = np.argmax(ucb)
        rewards[index] += int(R[index,i])
        select[index] += 1 
    #w = select/sum(select)
    w[np.argmax(select)] = 1
    '''
    R = context["R"]#120*25 len=120
    Nchoice = 500
    choice = np.random.random((Nchoice,n))
    w = np.zeros(n)
    ucb = np.zeros(Nchoice)
    select = np.zeros(Nchoice)
    rewards = np.zeros(Nchoice)

    N = 500
    for i in range(N):
        for j in range(Nchoice):
            if(select[j]==0):	
                ucb[j] = rewards[j]+1e400
            else:				
                ucb[j] = rewards[j]/select[j]+np.sqrt(3/2*np.log(i+1)/select[j])
        index = np.argmax(ucb)
        rewards[index] += int(np.dot(R[i%120,:],choice[index,:]).T)
        select[index] += 1 

    w = np.squeeze(np.asarray(np.dot(np.matrix(select),choice)))
    w /= sum(w)
    '''
    return w


if __name__ == "__main__":
    print("this is UCB Portfolio")
