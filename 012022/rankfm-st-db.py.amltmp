#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rankfm.rankfm import RankFM
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
import warnings


# ## Streamlit config

# In[ ]:


if 'total' not in st.session_state:
    st.session_state['total'] = 300


# In[ ]:


st.header("Telco Recs - Cable TV Packages recommender")
st.write("Recommendations predicteed using Factorization Machines (FM) ")


# In[ ]:


img=Image.open("data/fm_data/main.png")

st.image(img,width=800)


# ## Dataset Sourcing

# In[94]:


#Load data
path = "data/up-selling/peotv/"

#Ratings
ratings = pd.read_csv(path+'azure/peoTV_user_ratings.csv')

try:
    ratings.drop(["Unnamed: 0","ratings"],axis=1,inplace=True)
except:
    pass


# In[95]:


ITEM_COLUMN = "itemId"
USER_COULMN = "userId"
RATING_COLUMN = "rating"


# In[96]:


ratings.columns = [USER_COULMN, ITEM_COLUMN]

ratings.sort_values(by=[USER_COULMN],inplace=True,ascending=False)
# ratings = ratings.drop_duplicates(keep="first").reset_index().drop("index", axis=1)


# In[97]:


le = LabelEncoder()
ratings[ITEM_COLUMN+"_des"] = ratings[ITEM_COLUMN]
ratings[USER_COULMN+"_des"] = ratings[USER_COULMN]
ratings[USER_COULMN] = le.fit_transform(ratings[USER_COULMN])
ratings[ITEM_COLUMN] = le.fit_transform(ratings[ITEM_COLUMN])

r_dict = pd.Series(ratings[ITEM_COLUMN+"_des"].values,index=ratings[ITEM_COLUMN]).to_dict()
u_dict = pd.Series(ratings[USER_COULMN+"_des"].values,index=ratings[USER_COULMN]).to_dict()

final_r_dict = {}
final_u_dict = {}

for key,value in r_dict.items():
    if value not in final_r_dict.values():
        final_r_dict[key] = value

for key,value in u_dict.items():
    if value not in final_u_dict.values():
        final_u_dict[key] = value


# In[98]:


np.random.seed(100)
interactions_train, interactions_valid = np.split(ratings[[USER_COULMN,ITEM_COLUMN]], [int(.7*len(ratings))])


# In[99]:


interactions_train.shape


# In[100]:


interactions_valid.shape


# In[101]:


model = RankFM(factors=20, loss='warp', max_samples=20, alpha=0.01, sigma=0.1, learning_rate=0.1, learning_schedule='invscaling')
model.fit(interactions_train, epochs=20, verbose=True)


# In[102]:


valid_scores = model.predict(interactions_valid, cold_start='nan')
valid_scores


# In[103]:


valid_recs = model.recommend(ratings[USER_COULMN], n_items=10, filter_previous=True, cold_start='drop')


# In[104]:


def remap_labels(col):
    for k,v in final_r_dict.items():
        if int(col) == k:
            return v

def remap_index(col):
    for k,v in final_u_dict.items():
        if int(col) == k:
            return v


# In[105]:


valid_recs.drop_duplicates(inplace=True)
for col in list(valid_recs.columns):
    valid_recs[col] = valid_recs[col].apply(lambda x: remap_labels(x))
valid_recs.index = valid_recs.index.map(remap_index)


# In[106]:


valid_recs.head()


# In[107]:


valid_recs.to_csv("data/up-selling/peotv/fm/rank_fm_top_10_recs.csv")

