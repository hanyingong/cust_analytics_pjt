# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## using graphlab to implement recommend

# Step 0 : Import Libraries & Dataset for general user

import pandas as pd
import graphlab as gl

df = pd.read_csv('dataset\stacked_for_graphlab_without0.csv', sep = ',')
df.columns = ['Card_ID', '_SEGMENT_LABEL_', 'pdt_type', 'total_count']


###################### Segment 4 : Most Valuable ###########################

# Step 1: Prepare Datatset 
df_seg4_valuable = df[df._SEGMENT_LABEL_== 'Cluster4']  
df_seg4_valuable_SFrame = gl.SFrame(df_seg4_valuable)## convert into S-Frame
df_seg4_valuable_SFrame.save('dataset\dataset4.csv', format='csv') 

# Step 2: Split into test and training set

train_s4, test_s4 = gl.recommender.util.random_split_by_user (df_seg4_valuable_SFrame,\
                    user_id = 'Card_ID', item_id = 'pdt_type', \
                    item_test_proportion = 0.5, random_seed = 2017)

# Step 3: Create Model 1 - Pearson Similarity Score

train_s4_model_pearson = gl.recommender.item_similarity_recommender.create \
                (train_s4, user_id='Card_ID', item_id='pdt_type',\
                similarity_type='pearson')


# Step 4: Create Model 2 - Jaccard Similarity Score
train_s4_model_jaccard = gl.recommender.item_similarity_recommender.create \
                (train_s4, user_id='Card_ID', item_id='pdt_type',\
                similarity_type='jaccard')

# Step 5: Create Model 3 - Factorization 

train_s4_model_factorization = gl.recommender.ranking_factorization_recommender.\
                create(train_s4,\
                user_id='Card_ID', item_id='pdt_type',\
                 random_seed = 2017, solver = 'ials')
            
#Model comparsion 
x1 = gl.recommender.util.compare_models(test_s4 , \
    [train_s4_model_pearson, train_s4_model_jaccard, train_s4_model_factorization ], model_names=["m1", "m2", "m3"])
    
## Selection of Model and run on whole dataset
    
train_s4_model_final= gl.recommender.item_similarity_recommender.create \
                (df_seg4_valuable_SFrame, user_id='Card_ID', item_id='pdt_type',\
                similarity_type='pearson')

recs_final  = train_s4_model_final.recommend()
recs_final.save('dataset\dataset_final_recs4.csv', format='csv')
