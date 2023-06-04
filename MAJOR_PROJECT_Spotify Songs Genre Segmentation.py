#!/usr/bin/env python
# coding: utf-8

# # Perform data pre-processing operations.

# In[91]:


import pandas as pd
import numpy as np


# In[92]:


df = pd.read_csv("spotify dataset.csv")
df


# In[93]:


df.describe()


# In[94]:


df.head()


# In[95]:


df.tail()


# In[97]:


df.isnull().values.any()


# In[98]:


df.isnull().mean()


# # Label Encoding on text/Categorical data

# In[46]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['track_name']= label_encoder.fit_transform(df['track_name'])

df['track_name'].unique()


df['track_artist']= label_encoder.fit_transform(df['track_artist'])

df['track_artist'].unique()


df['track_album_name']= label_encoder.fit_transform(df['track_album_name'])

df['track_album_name'].unique()

df['track_id']= label_encoder.fit_transform(df['track_id'])

df['track_id'].unique()


df['track_album_id']= label_encoder.fit_transform(df['track_album_id'])

df['track_album_id'].unique()


df['track_album_release_date']= label_encoder.fit_transform(df['track_album_release_date'])

df['track_album_release_date'].unique()


df['playlist_name']= label_encoder.fit_transform(df['playlist_name'])

df['playlist_name'].unique()



df['playlist_id']= label_encoder.fit_transform(df['playlist_id'])

df['playlist_id'].unique()


df['playlist_genre']= label_encoder.fit_transform(df['playlist_genre'])

df['playlist_genre'].unique()


# In[47]:


df


# In[48]:


df.isnull().mean()


# # data analysis and visualizations draw all the possible plots to provide essential informations and to derive some meaningful insights.

# In[49]:


import matplotlib.pyplot as plt
x=df["track_popularity"]
y=df["duration_ms"]
plt.scatter(x, y, c='r')
plt.xlabel("track_popularity")
plt.ylabel("duration_ms")
plt.title("popularity vs duration")
plt.show()


# In[12]:


import matplotlib.pyplot as plt
x=df["playlist_genre"]
y=df["duration_ms"]
plt.bar(x, y)
plt.xlabel("playlist_genre")
plt.ylabel("duration_ms")
plt.title("playlist_genre vs duration")
plt.show()


# In[13]:


import matplotlib.pyplot as plt
x=df["playlist_genre"]
y=df["duration_ms"]
plt.hist(x, bins=30)
plt.xlabel("playlist_genre")
plt.ylabel("duration_ms")
plt.title("playlist_genre vs duration")
plt.show()


# In[15]:


import seaborn as sns
sns.lineplot(x="playlist_id",y="duration_ms",data = df)
plt.xlabel('playlist_id')
plt.ylabel('duration_ms')
plt.title("playlist_id vs duration_ms")
plt.show()


# In[16]:


import seaborn as sns
sns.lineplot(x="track_album_release_date",y="duration_ms",data = df)
plt.xlabel('track_album_release_date')
plt.ylabel('duration_ms')
plt.title("track_album_release_date vs duration_ms")
plt.show()


# In[22]:


import seaborn as sns
sns.histplot(x="speechiness",y="duration_ms",data = df)
plt.xlabel('speechiness')
plt.ylabel('duration_ms')
plt.title("speechiness vs duration_ms")
plt.show()


# In[23]:


import seaborn as sns
sns.lmplot(x="track_artist",y="duration_ms",data = df)
plt.xlabel('track_artist')
plt.ylabel('duration_ms')
plt.title("track_artist vs duration_ms")
plt.show()


# In[24]:


import seaborn as sns
sns.scatterplot(x="danceability",y="duration_ms",data = df)
plt.xlabel('danceability')
plt.ylabel('duration_ms')
plt.title("danceability vs duration_ms")
plt.show()


# In[25]:


import seaborn as sns
sns.barplot(x="acousticness",y="duration_ms",data = df)
plt.xlabel('acousticness')
plt.ylabel('duration_ms')
plt.title("acousticness vs duration_ms")
plt.show()


# # correlation matrix of features according to the datasets.

# In[50]:


import pandas as pd
import seaborn as sns

corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)


# In[54]:


df


# # K-MEANS CLUSTERING

# In[67]:


from sklearn.cluster import KMeans


# In[69]:


km=KMeans(n_clusters=5)
km.fit(df[["track_id","track_name","track_artist","track_popularity","track_album_id","track_album_name","track_album_release_date","playlist_name","playlist_id","playlist_genre","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_ms"]])


# In[70]:


km.cluster_centers_


# In[71]:


df["cluster_group"]=km.labels_


# In[72]:


df


# # plot different clusters according to different parameters like playlist genres , playlist names.

# In[87]:


sns.scatterplot(x="playlist_genre",y="playlist_name",data=df,hue="cluster_group")


# In[76]:


sns.scatterplot(x="playlist_name",y="playlist_genre",data=df,hue="cluster_group")


# In[77]:


sns.scatterplot(x="playlist_id",y="playlist_name",data=df,hue="cluster_group")


# In[86]:


sns.scatterplot(x="playlist_id",y="duration_ms",data=df,hue="cluster_group")


# In[81]:


sns.scatterplot(x="track_id",y="track_popularity",data=df,hue="cluster_group")


# In[85]:


sns.scatterplot(x="track_album_id",y="duration_ms",data=df,hue="cluster_group")


# In[89]:


sns.scatterplot(x="track_album_release_date",y="duration_ms",data=df,hue="cluster_group")


# In[ ]:




