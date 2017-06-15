# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

## using graphlab to implement recommend

# Step 0 : Import Libraries & Dataset for general user

import pandas as pd
import graphlab as gl

df = pd.read_csv('dataset\dataset02_master.csv', sep = ',')

print(df.head(5)) ## check data

###################### Segment 2 : Most Growable ##############################

# Step 1: Prepare Datatset 
df_seg2_growable = df[df.SegmentNo == 2]  
df_seg2_growable = df_seg2_growable[['Card_ID', 'pdt_type']]
df_seg2_growable_SFrame = gl.SFrame(df_seg2_growable)## convert into S-Frame
df_seg2_growable_SFrame.save('dataset\dataset_recommender_2.csv', format='csv') 

# Step 2: Split into test and training set

train_s2, test_s2 = gl.recommender.util.random_split_by_user (df_seg2_growable_SFrame,\
                    user_id = 'Card_ID', item_id = 'pdt_type', \
                    item_test_proportion = 0.5, random_seed = 2017)

# Step 3: Create Model 1 - Pearson Similarity Score

train_s2_model_pearson = gl.recommender.item_similarity_recommender.create \
                (train_s2, user_id='Card_ID', item_id='pdt_type',\
                similarity_type='pearson')


# Step 4: Create Model 2 - Jaccard Similarity Score
train_s2_model_jaccard = gl.recommender.item_similarity_recommender.create \
                (train_s2, user_id='Card_ID', item_id='pdt_type',\
                similarity_type='jaccard')

# Step 5: Create Model 3 - Factorization 

train_s2_model_factorization = gl.recommender.ranking_factorization_recommender.\
                create(train_s2,\
                user_id='Card_ID', item_id='pdt_type',\
                 random_seed = 2017, solver = 'ials')
            
#Model comparsion 
x2 = gl.recommender.util.compare_models(test_s2 , \
    [train_s2_model_pearson, train_s2_model_jaccard, train_s2_model_factorization ], model_names=["m1", "m2", "m3"])
    
## Selection of Model and run on whole dataset
    
train_s2_model_final= gl.recommender.item_similarity_recommender.create \
                (df_seg2_growable_SFrame, user_id='Card_ID', item_id='pdt_type',\
                similarity_type='jaccard')

recs_final  = train_s2_model_final.recommend()
recs_final.save('dataset\dataset_final_recs2.csv', format='csv')




###################### Segment 4 : Most Valuable ##############################

# Step 1: Prepare Datatset 
df_seg4_valuable = df[df.SegmentNo == 4]  
df_seg4_valuable = df_seg4_valuable[['Card_ID', 'pdt_type']]
df_seg4_valuable_SFrame = gl.SFrame(df_seg4_valuable)## convert into S-Frame
df_seg4_valuable_SFrame.save('dataset\dataset_recommender_4.csv', format='csv') 

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
x4 = gl.recommender.util.compare_models(test_s4 , \
    [train_s4_model_pearson, train_s4_model_jaccard, train_s4_model_factorization ], model_names=["m1", "m2", "m3"])
    
## Selection of Model and run on whole dataset
    
train_s4_model_final= gl.recommender.item_similarity_recommender.create \
                (df_seg4_valuable_SFrame, user_id='Card_ID', item_id='pdt_type',\
                similarity_type='jaccard')

recs_final  = train_s4_model_final.recommend()
recs_final.save('dataset\dataset_final_recs4.csv', format='csv')





###################### Segment 1 : churn_valuable ##############################

# Step 1: Prepare Datatset 
df_seg1_churn_valuable = df[df.SegmentNo == 1]  
df_seg1_churn_valuable  = df_seg1_churn_valuable[['Card_ID', 'pdt_type']]
df_seg1_churn_valuable_SFrame = gl.SFrame(df_seg1_churn_valuable)## convert into S-Frame
df_seg1_churn_valuable_SFrame.save('dataset\dataset_recommender_1.csv', format='csv') 

# Step 2: Split into test and training set

train_s1, test_s1 = gl.recommender.util.random_split_by_user (df_seg1_churn_valuable_SFrame,\
                    user_id = 'Card_ID', item_id = 'pdt_type', \
                    item_test_proportion = 0.5, random_seed = 2017)

# Step 3: Create Model 1 - Pearson Similarity Score

train_s1_model_pearson = gl.recommender.item_similarity_recommender.create \
                (train_s1, user_id='Card_ID', item_id='pdt_type',\
                similarity_type='pearson')


# Step 4: Create Model 2 - Jaccard Similarity Score
train_s1_model_jaccard = gl.recommender.item_similarity_recommender.create \
                (train_s1, user_id='Card_ID', item_id='pdt_type',\
                similarity_type='jaccard')

# Step 5: Create Model 3 - Factorization 

train_s1_model_factorization = gl.recommender.ranking_factorization_recommender.\
                create(train_s1,\
                user_id='Card_ID', item_id='pdt_type',\
                 random_seed = 2017, solver = 'ials')
            
#Model comparsion 
x1 = gl.recommender.util.compare_models(test_s4 , \
    [train_s1_model_pearson, train_s1_model_jaccard, train_s1_model_factorization ], model_names=["m1", "m2", "m3"])
    
## Selection of Model and run on whole dataset
    
train_s1_model_final= gl.recommender.item_similarity_recommender.create \
                (df_seg1_churn_valuable_SFrame, user_id='Card_ID', item_id='pdt_type',\
                similarity_type='jaccard')

recs_final  = train_s1_model_final.recommend()
recs_final.save('dataset\dataset_final_recs1.csv', format='csv')

