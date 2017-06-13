# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 14:46:50 2017

@author: hanying.ong.2015
"""

## CA Project 

## Step 0: import library

import pandas as pd
import numpy as np

## Step 1 : download dataset - original dataset

top_20pc_ds = pd.read_csv('dataset/Top2CLV_Customers.csv', sep =',')


## Step 2: Set parameters & mehodologys
from sklearn.metrics import jaccard_similarity_score


## Step 3: Determine Variables

top_20pc_ds_cust_pdt = top_20pc_ds[['Card_ID','Entertain_low', 'Entertain_medium', 'Entertainment_high', 'Game_low',
       'Game_medium', 'Game_high', 'Hardware_low', 'Hardware_medium',
       'Hardware_high', 'HiFi_low', 'HiFi_medium', 'HiFi_high',
       'MP3Players_low', 'MP3Players_medium', 'MP3Players_high',
       'Software_low', 'Software_medium', 'Software_high', 'Telephony_low',
       'Telephony_medium', 'Telephony_high']]


pdt_table = ['Entertain_low', 'Entertain_medium', 'Entertainment_high', 'Game_low',
       'Game_medium', 'Game_high', 'Hardware_low', 'Hardware_medium',
       'Hardware_high', 'HiFi_low', 'HiFi_medium', 'HiFi_high',
       'MP3Players_low', 'MP3Players_medium', 'MP3Players_high',
       'Software_low', 'Software_medium', 'Software_high', 'Telephony_low',
       'Telephony_medium', 'Telephony_high']

pdt_row_no = (np.max(top_20pc_ds_cust_pdt.iloc[0, 1:], axis = 0))-1
pdt_row_name = pdt_table[pdt_row_no] 


top_20pc_ds_pdt = top_20pc_ds[['Entertain_low', 'Entertain_medium', 'Entertainment_high', 'Game_low',
       'Game_medium', 'Game_high', 'Hardware_low', 'Hardware_medium',
       'Hardware_high', 'HiFi_low', 'HiFi_medium', 'HiFi_high',
       'MP3Players_low', 'MP3Players_medium', 'MP3Players_high',
       'Software_low', 'Software_medium', 'Software_high', 'Telephony_low',
       'Telephony_medium', 'Telephony_high']]

## return customer preference via total count (Top 1 first)

#cust_id = []
#cust_id_rank = []
#for i in range (top_20pc_ds_cust_pdt.shape[0]):
#    temp_cust_id_rank = []
#    temp_cust_id = top_20pc_ds_cust_pdt.iloc[i,0]
#    
#    pdt_row_no = (np.max(top_20pc_ds_cust_pdt.iloc[i, 1:], axis = 0))-1
#    pdt_row_name = pdt_table[pdt_row_no] 


## Step 4: Splitting the data into train and test 

## Step 5: test on cosine score

## step 6: test on jaccard score

score = []
score_details = []

for i in range (0, top_20pc_ds_pdt.shape[1]): # 7 catagories 
    score_details = []
    
    for j in range (0, top_20pc_ds_pdt.shape[1]):
        x = (i - j)
        temp = jaccard_similarity_score(top_20pc_ds_pdt.iloc[:,i], top_20pc_ds_pdt.iloc[:,i-x], \
                                        normalize=True, sample_weight=None)
        score_details.append(temp)  
    score.append(score_details)

top_20pc_jaccard_score= pd.DataFrame(np.array(score))
top_20pc_jaccard_score.columns = ['Entertain_low', 'Entertain_medium', 'Entertainment_high', 'Game_low',
       'Game_medium', 'Game_high', 'Hardware_low', 'Hardware_medium',
       'Hardware_high', 'HiFi_low', 'HiFi_medium', 'HiFi_high',
       'MP3Players_low', 'MP3Players_medium', 'MP3Players_high',
       'Software_low', 'Software_medium', 'Software_high', 'Telephony_low',
       'Telephony_medium', 'Telephony_high']

top_20pc_jaccard_score.index = ['Entertain_low', 'Entertain_medium', 'Entertainment_high', 'Game_low',
       'Game_medium', 'Game_high', 'Hardware_low', 'Hardware_medium',
       'Hardware_high', 'HiFi_low', 'HiFi_medium', 'HiFi_high',
       'MP3Players_low', 'MP3Players_medium', 'MP3Players_high',
       'Software_low', 'Software_medium', 'Software_high', 'Telephony_low',
       'Telephony_medium', 'Telephony_high']

## step 7: test on latent factor

## step 8: comparing the score

## final model selection: 
    
    