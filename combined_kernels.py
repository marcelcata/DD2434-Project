import numpy as np

#Combination of 2 kernel Gram Matrices
def combine2(Gram_1, Gram_2, w1, w2):
    n_rows = np.size(Gram_1,0)
    n_cols = np.size(Gram_1,1)

    Combined_Gram_Matrix = np.zeros((n_rows,n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            Combined_Gram_Matrix[i][j] = Gram_1[i][j]*w1 + Gram_2[i][j]*w2

    return Combined_Gram_Matrix


def combine3(Gram_1, Gram_2, Gram_3, w1, w2, w3):
    n_rows = np.size(Gram_1,0)
    n_cols = np.size(Gram_1,1)

    Combined_Gram_Matrix = np.zeros((n_rows,n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            Combined_Gram_Matrix[i][j] = Gram_1[i][j]*w1 + Gram_2[i][j]*w2 + Gram_3[i][j]*w3

    return Combined_Gram_Matrix