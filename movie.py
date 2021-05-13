#!/usr/bin/env python
# coding: utf-8

# # Gravitational Force based movie recommendation

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st


# In[2]:


from sklearn.feature_extraction.text import CountVectorizer


# In[3]:


movies=pd.read_csv('movies.csv')
tags=pd.read_csv('tags.csv')
ratings=pd.read_csv('ratings.csv')
links=pd.read_csv('links.csv')
move=pd.read_csv('movies.csv')


# In[4]:


name= move['title'].tolist()
name=name[:500]
len(name)


# In[5]:


weights=np.log(ratings.groupby('movieId').rating.count())


# In[6]:


move['genres']=move.genres.str.split('|')
vect= CountVectorizer()
vect.fit_transform(movies.genres)
vect.get_feature_names()
vect_matrix=vect.transform(movies.genres)


# In[7]:


from sklearn.metrics.pairwise import cosine_similarity
tag_sim=cosine_similarity(vect_matrix[0:9742], vect_matrix)
g_sim=[]
move.set_index(move.movieId,inplace=True)
move['weights']=np.log(ratings.groupby('movieId').rating.count())
move['index']=range(9742)
move.set_index('index',inplace=True)


# In[8]:


for i in range(500):
    temp=[]
    for j in range(500):
        dist=1-tag_sim[i][j]
        if(dist==0):
            temp.append(1)
        else:
            if(i==j):
                  temp.append(1)
            else:
                 x=(move.weights[i]*move.weights[j])/(dist**2)
                 temp.append(x)
    g_sim.append(temp)
   #print(len(g_sim))


# In[9]:


st.title("Movie Recommendation System")
from PIL import Image
img=Image.open("C:/Users/HP/Desktop/film.jpg")
st.image(img,width=300)

film=st.text_input("Enter your movie name")
val = st.slider("Similarity Score",0,200,key = "<uniquevalueofsomesort>")

similar_movie={}
for i in range(len(name)):
    similar_movie[name[i]]=[]
    for j in range(len(g_sim[i])):
        if g_sim[i][j]>=val:
            similar_movie[name[i]].append(name[j])
#print(similar_movie)


# In[12]:


Recommended_movie=""
if st.button("Recommend"):
    Recommended_movie=similar_movie[film]
    st.success(f"Your Recommended movies are:{Recommended_movie}")



# In[ ]:
