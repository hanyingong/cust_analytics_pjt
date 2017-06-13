# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 10:29:24 2017

@author: hanying.ong.2015
"""

import pandas as pd
import numpy as np
import time

# Phase 1: Prepare the Pivot of Data  - Analaysis at Product Category
trans_data = pd.read_csv('dataset/tb1_trans_pdt.csv', sep = ',')

trans_data['Card_ID'] = trans_data.Card_ID.astype('object')
trans_data['N'] = 1

#Product_Category_Level Only
trans_data_pdt_cat = pd.pivot_table(trans_data, values = 'N', \
                                  index = ['Card_ID'], \
                                  columns = ['Product_Category'],\
                                  aggfunc = np.sum, fill_value = 0)

## Note : no purchase == disinterest

# Phase 2: Latent Factor 

###############################################################################

# Source : http://www.quuxlabs.com/blog/2010/09/
#           matrix-factorization-a-simple-tutorial-and-implementation-in-python/#source-code
""" 
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
    
print ('----------------------start----------------------')
print('start def factorization' )
start_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print('Start Time: ' , start_time)
print('')

"""
def matrix_factorization(R, P, Q, K, steps=100, alpha=0.0002, beta=0.02): #steps = 1000
    Q = Q.T
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P,Q)
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

###############################################################################

#to align back to index later on

trans_data_pdt_cat['Card_ID'] = trans_data_pdt_cat.index
trans_data_pdt_cat = trans_data_pdt_cat.reset_index(drop = True)

#convert to np.array to use the function above

R = np.array(trans_data_pdt_cat.drop('Card_ID', axis = 1))

N = len(R)
M = len(R[0])
K = 5 #per in class

P = np.random.rand(N,K) #random latent factor
Q = np.random.rand(M,K) #random latent factor

print ('----------------------start----------------------')
print('start matrix factorization' )
start_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print('Start Time: ' , start_time)
print('')


nP, nQ = matrix_factorization(R, P, Q, K)

print ('----------------------start----------------------')
print('start matrix factorization - np.dot' )
start_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print('Start Time: ' , start_time)
print('')


nR = np.dot(nP, nQ.T)

end_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print('End Time: ' , end_time)
print ('--------------------completed--------------------')
print("")

nR_round = np.round(nR, 2)

# convert it back to dataframe for comparison later

data_pdt_cat_LF = pd.DataFrame(nR)

data_pdt_cat_LF.columns = ['Entertainment', 'GameConsoles', 'Hardware', \
                           'HiFi', 'MP3Players', 'Software', 'Telephony']

data_pdt_cat_LF['Card_ID'] = trans_data_pdt_cat['Card_ID'] 



end_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
print('End Time: ' , end_time)
print ('--------------------completed--------------------')
print("")

# Phase 3: Jaccard Score 

from sklearn.metrics import jaccard_similarity_score 

# Phase 4: Cosine Similarity Score


# Phase 5: Evaluation - Comparing the Precision Score for all 3 methods

# Phase 6: Final Recommendation




