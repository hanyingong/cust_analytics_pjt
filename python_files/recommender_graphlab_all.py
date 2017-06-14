# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:36:35 2017

@author: hanying.ong.2015
"""

# Step 0 : Import Libraries & Dataset for general user

import pandas as pd
import graphlab as gl

df = pd.read_csv('dataset\dataset02_master.csv', sep = ',')

print(df.head(5)) ## check data

###################### Segment ALL ##############################

# Step 1: Prepare Datatset 

## Set 1, dataset for recommendation filtering
df_all = df[['Card_ID','pdt_type']].astype(str)

## Set 2, dataset for user info
df_user_data = df[['Card_ID','Gender', 'Age', 'Age_Grp',
       'Length_of_Membership_MTH', 'Membership_Grp','Type']].\
       drop_duplicates().reset_index(drop = True)       
df_user_data.astype(str)

## convert into S-Frame
df_all_SFrame = gl.SFrame(df_all)
df_user_data_SFrame = gl.SFrame(df_user_data)


# Step 3: Create Recommendation Models - Autoselection of best match
all_model = gl.recommender.create \
                (df_all_SFrame, user_id='Card_ID', item_id='pdt_type',\
                user_data=df_user_data_SFrame)

recs_final  = all_model.recommend()
results_final= recs_final.to_dataframe()

## keep only customers from cluster 7 : New customers

df_user_data_7 = df[df.SegmentNo == 7].astype(str)
results_final_7 = df_user_data_7['Card_ID'].isin(results_final['Card_ID'])

results_final_7 = results_final[results_final['Card_ID'].\
                                       isin(df_user_data_7['Card_ID'])].\
                                       reset_index(drop = True)
                                       
## output into csv file for further analysis & visualization in tableau
df_user_data_7.to_csv('dataset\dataset_final_all_user.csv', format='csv')
results_final_7.to_csv('dataset\dataset_final_all.csv', format='csv')
