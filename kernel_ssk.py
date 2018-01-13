# SSK KERNEL
#Efficient Computation of SSK
import numpy as np
import math

#Compute kernel for strings s and t
def compute(s,t,k,m_lambda):
    Kst= computeK(s,t,k,m_lambda)
    Kss= computeK(s,s,k,m_lambda)
    Ktt= computeK(t,t,k,m_lambda)
    denominator = math.sqrt(Kss*Ktt) + 10e-20
    return Kst / denominator


def computeK(s,t,k,m_lambda):
    len_s = len(s)
    len_t = len(t)
    max_len = k  # max lenght
    Kp = np.zeros([max_len + 1, len_s, len_t])  # kp = k'
    Kpp = 0.0  # Kpp = k''

    # initialize for 0 subsequence length for both the strings
    for i in range(len_s):
        for j in range(len_t):
            Kp[0][i][j] = 1.0

    # computing of the K' (Kp) function using equations shown in Lodhi et. al.
    for i in range(max_len):
        for j in range(len_s - 1):
            Kpp = 0.0
            for x in range(len_t - 1):
                Kpp = m_lambda * (Kpp + m_lambda * (s[j] == t[x]) * Kp[i][j][x])
                Kp[i + 1][j + 1][x + 1] = m_lambda * Kp[i + 1][j][x + 1] + Kpp

    # compute the kernel function
    K = 0.0;
    for i in range(max_len):
        for j in range(len_s):
            for x in range(len_t):
                K += m_lambda * m_lambda * (s[j] == t[x]) * Kp[i][j][x]

    return K;
