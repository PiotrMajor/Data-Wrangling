#!/usr/bin/env python
# coding: utf-8

# # Wrangle and Analyze Data Project
# 
# ## Introduction
# 
# The purpose of this project is to wrangle data about WeRateDogs Twitter account. First, data is going to be obtained from multiple sources. Next, the data is going to be assessed in terms of both quality and tidiness (https://ryanwingate.com/purpose/tidy-data/ ). Finally, I will clean the data and perform simple analysis.

# ## Gathering Data

# In[1]:


# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import requests
import os


# In[2]:


# load Twitter archive data and display 1 row
df_archive_original = pd.read_csv('twitter-archive-enhanced.csv')
df_archive_original.head(1)


# In[3]:


# Download the tweet image predictions
url = 'https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv'
local_filename = url.split('/')[-1]
r = requests.get(url)
f = open(local_filename, 'wb')
f.write(r.content)


# In[4]:


# open and look at the predictions
df_pred_original = pd.read_csv('image-predictions.tsv', sep='\t')
df_pred_original.head()


# In[5]:


# Check for tweet_json.txt
# if tweet_json.txt doesn't exist, download programatically
url = 'https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be5fb7d_tweet-json/tweet-json.txt'
json_txt_file_path = 'tweet_json.txt'
if not os.path.isfile(json_txt_file_path):
    with open(json_txt_file_path, mode = 'wb') as file:
        file.write(requests.get(url).content)
# Load into DataFrame
df_json_original = pd.read_json(json_txt_file_path, lines=True)
df_json_original = df_json_original[['id', 'retweet_count', 'favorite_count']]
df_json_original.head()


# Make copies of the dataframes for the analysis.

# In[6]:


df_archive = df_archive_original.copy()
df_pred = df_pred_original.copy()
df_json = df_json_original.copy()


# ## Assessing Data

# In[7]:


df_archive.head(1)


# In[8]:


df_pred.head(5)


# In[9]:


df_json.head(1)


# In[10]:


df_archive.info()


# In[11]:


df_pred.info()


# In[12]:


df_json.info()


# In[13]:


df_archive.rating_denominator.value_counts()


# In[14]:


df_archive.name.value_counts()


# ### Quality
# 
# 1. Change the **id** column in `df_json` dataframe to **tweet_id** in order to match it with other dataframes. This will make is easier to merge dataframes at the last steps. In addition, I will change the data type from int to string because ids should not be numeric and they aren't intended to perform calculations.
# 2. Change **timestamp** data type in `df_archive`. This step will help if I was to analyze the data based on time or date.
# 3. Drop **retweets** rows in `df_archive`. As per project instructions, only original tweets must be included in the final master dataset.
# 4. Drop unnecessary columns in `df_archive`. It is not properly an issue, but it will make it easier to analyze the data. That is why I have 9 bullet points in this section.
# 5. Drop null values from **expanded_urls** in `df_archive`. The **expanded_urls** is the only column that has a few missing values. I could not determine why that was the case nor I could fill these values out; therefore, I will drop these values.
# 6. Drop tweets that are not dogs. This analysis should only incorporate tweets of dogs; therefore, the next step is to delete rows that contain non-dog tweets. In order to do that, I will merge the df_pred dataframe with df_archive and create a new dataframe df_archive2. The df_pred contains 3 predictions from the neural network whether a picture shows a dog. I assume that if any of the predictions suggest that it is a dog, I treated it as a dog. I checked a few tweets ‘by hand’ and it appeared to be true.
# 7. Clean **rating_denominator** in `df_archive`. Extract the values from text, delete tweets without ratings and tweets with multiple dogs. Change the data type to float.
# 8. Clean **rating_nominator** in `df_archive`. Extract the values from text, delete tweets without ratings and tweets with multiple dogs. Change the data type to float.
# 9. Clean names using regular expression in `df_archive`. A useful pattern to find these "fake names" is that they started with lowercase, instead, the "real names" always started with uppercase, moreover, the real ones usually are placed after words like "named", "is" etc...so it'is possible to clean this issue using a regex.
# 

# ### Tidiness
# 1. Melt 'doggo' 'floofer' 'pupper' 'puppo' in `df_archive`.
# 2. Join all data frames into `twitter_archive_master.csv`.

# ## Cleaning Data
# ### Quality

# **Define**
# 1. Change the **id** column in `df_json` dataframe to **tweet_id** in order to match it with other dataframes. This will make is easier to merge dataframes at the last steps. In addition, I will change the data type from int to string because ids should not be numeric and they aren't intended to perform calculations.

# **Code**

# In[15]:


# rename to `tweet_id`
df_json = df_json.rename(columns={'id': 'tweet_id'})


# In[16]:


# change `tweet_id` datatype to string in all 3 data frames
df_archive['tweet_id']= df_archive['tweet_id'].astype(str)
df_pred['tweet_id']= df_pred['tweet_id'].astype(str)
df_json['tweet_id']= df_json['tweet_id'].astype(str)


# **Test**

# In[17]:


df_json.head(1)


# In[18]:


df_json.info()


# **Define**
# 2. Change **timestamp** data type in `df_archive`. This step will help if I was to analyze the data based on time or date.

# **Code**

# In[19]:


# change data type to datetime
df_archive['timestamp'] = pd.to_datetime(df_archive.timestamp)


# **Test**

# In[20]:


df_archive.info()


# **Define**
# 3. Drop **retweets** rows in `df_archive`. As per project instructions, only original tweets must be included in the final master dataset.

# **Code**

# In[21]:


# check how many retweets are in the dataframe
df_archive.info()


# There are 181 retweets in the dataframe. Next, I will delete the retweets.

# In[22]:


# select retweets by checking the retweets columns (if they are empty it's an original tweet).
df_archive_notnull = df_archive[df_archive.retweeted_status_id.notnull()]
df_archive_notnull


# In[23]:


# drop retweets (df_archive.retweeted_status rows that are not empty)
df_archive = df_archive.drop(df_archive_notnull.index, axis=0)
df_archive


# **Test**

# In[24]:


# confirm changes (retweeted_status_id should be 0)
df_archive.info()


# **Define**
# 4. Drop unnecessary columns in `df_archive`. It is not properly an issue, but it will make it easier to analyze the data. That is why I have 9 bullet points in this section.

# **Code**

# In[25]:


# drop columns
df_archive.drop(['in_reply_to_status_id','in_reply_to_user_id','retweeted_status_id', 'retweeted_status_user_id', 'retweeted_status_timestamp', 'source'], axis=1, inplace = True)


# **Test**

# In[26]:


# confirm changes
df_archive


# **Define**
# 5. Drop null values from **expanded_urls** in `df_archive`. The **expanded_urls** is the only column that has a few missing values. I could not determine why that was the case nor I could fill these values out; therefore, I will drop these values.

# **Code**

# In[27]:


# drop null values
df_archive.dropna(inplace=True)


# **Test**

# In[28]:


# confirm changes
df_archive.info()


# **Define**
# 6. Drop tweets that are not dogs. This analysis should only incorporate tweets of dogs; therefore, the next step is to delete rows that contain non-dog tweets. In order to do that, I will merge the df_pred dataframe with df_archive and create a new dataframe df_archive2. The df_pred contains 3 predictions from the neural network whether a picture shows a dog. I assume that if any of the predictions suggest that it is a dog, I treated it as a dog. I checked a few tweets ‘by hand’ and it appeared to be true.

# **Code**

# In[29]:


# merge predictions into the archive
df_archive2 = df_archive.merge(df_pred, on=['tweet_id'], how='left')


# In[30]:


# check if the dataframes were merged correctly
df_archive2.info()


# In[31]:


# select pictures if any of the predictions suggest that it is a dog
df_archive2 = df_archive2.query('p1_dog == True | p2_dog == True | p3_dog == True')


# **Test**

# In[32]:


# check how many dog tweets are left
df_archive2.info()


# **Define**
# 7. Clean **rating_denominator** in `df_archive`. Extract the values from text, delete tweets without ratings and tweets with multiple dogs. Change the data type to float.

# 8. Clean **rating_denominator** in `df_archive`. Extract the values from text, delete tweets without ratings and tweets with multiple dogs. Change the data type to float.

# **Code**

# In[33]:


# change column width to view full text
pd.options.display.max_colwidth = 200


# In[34]:


# import regular expression
import re


# In[35]:


# extract ratings from the text column into a new column 'ratings'.
# Some ratings need to be fixed manually because there are multiple xx/xx occurrences in text
# or there are multiple dogs in one picture
df_archive2['ratings'] = df_archive2['text'].str.extract(r'(\d+\.?\d*/\d+){1}', expand = False)


# In[36]:


# Extract rating numerators from the new column 'ratings' as float
df_archive2['rating_numerator'] = df_archive2['ratings'].str.split('/').str.get(0).astype(float)


# In[37]:


# Extract rating denominators from the new column 'ratings' as float
df_archive2['rating_denominator'] = df_archive2['ratings'].str.split('/').str.get(1).astype(float)


# In[38]:


# drop ratings column after the extraction
df_archive2.drop(['ratings'], axis=1, inplace=True)


# In this section, I'm going to manually clean incorrecntly extracted ratings. I will delete tweets with multiple dogs.

# In[39]:


df_archive2['rating_denominator'].value_counts()


# In[40]:


df_archive2[df_archive2.rating_denominator == 70]


# In[41]:


# drop row 341 because it is multiple dogs
df_archive2.drop(341, inplace=True)


# In[42]:


df_archive2[df_archive2.rating_denominator == 7]


# In[43]:


# drop row 405 as there is no rating provided
df_archive2.drop(405, inplace=True)


# In[44]:


df_archive2[df_archive2.rating_denominator == 20]


# In[45]:


df_archive2.loc[953, 'rating_denominator'] = 10
df_archive2.loc[953, 'rating_numerator'] = 13


# In[46]:


df_archive2[df_archive2.rating_denominator == 40]


# In[47]:


# drop row 1218 because it is multiple dogs
df_archive2.drop(1218, inplace=True)


# In[48]:


df_archive2[df_archive2.rating_denominator == 90]


# In[49]:


# drop row 1016 because it is multiple dogs
df_archive2.drop(1016, inplace=True)


# In[50]:


df_archive2[df_archive2.rating_denominator == 110]


# In[51]:


# drop row 1411 because it is multiple dogs
df_archive2.drop(1411, inplace=True)


# In[52]:


df_archive2[df_archive2.rating_denominator == 120]


# In[53]:


# drop row 1552 because it is multiple dogs
df_archive2.drop(1552, inplace=True)


# In[54]:


df_archive2[df_archive2.rating_denominator == 130]


# In[55]:


# drop row 1410 because it is multiple dogs
df_archive2.drop(1410, inplace=True)


# In[56]:


df_archive2[df_archive2.rating_denominator == 150]


# In[57]:


# drop row 1410 because it is multiple dogs
df_archive2.drop(702, inplace=True)


# In[58]:


df_archive2[df_archive2.rating_denominator == 2]


# In[59]:


# change row 2096
df_archive2.loc[2096, 'rating_denominator'] = 10
df_archive2.loc[2096, 'rating_numerator'] = 9


# In[60]:


df_archive2[df_archive2.rating_denominator == 11]


# In[61]:


df_archive2.loc[857, 'rating_denominator'] = 10
df_archive2.loc[857, 'rating_numerator'] = 14


# In[62]:


df_archive2.loc[1438, 'rating_denominator'] = 10
df_archive2.loc[1438, 'rating_numerator'] = 10


# In[63]:


df_archive2[df_archive2.rating_denominator == 80]


# In[64]:


# drop rows 1041 and 1615 because it is multiple dogs
df_archive2.drop([1041,1615], inplace=True)


# In[65]:


df_archive2[df_archive2.rating_denominator == 50]


# In[66]:


df_archive2.loc[990, 'rating_denominator'] = 10
df_archive2.loc[990, 'rating_numerator'] = 11


# In[67]:


# drop rows 1061 and 1136 because it is multiple dogs
df_archive2.drop([1061,1136], inplace=True)


# **Clean Rating Numerator**

# In[68]:


df_archive2.rating_numerator.value_counts()


# In[69]:


df_archive2[df_archive2.rating_numerator == 0]


# In[70]:


df_archive2[df_archive2.rating_numerator == 2]


# In[71]:


df_archive2[df_archive2.rating_numerator == 3]


# In[72]:


df_archive2[df_archive2.rating_numerator == 4]


# In[73]:


df_archive2[df_archive2.rating_numerator == 5]


# In[74]:


df_archive2[df_archive2.rating_numerator == 6]


# In[75]:


df_archive2[df_archive2.rating_numerator == 7]


# In[76]:


df_archive2[df_archive2.rating_numerator == 13.5]


# **Test**

# In[77]:


# confirm that all denominators are 10
df_archive2['rating_denominator'].value_counts()


# In[78]:


# check for changed data type (float for ratings)
df_archive2.info()


# In[79]:


# confirm changes
df_archive2.rating_numerator.value_counts()


# All rating numerators look clean.

# **Define**
# 8. Clean names using regular expression in `df_archive`. A useful pattern to find these "fake names" is that they started with lowercase, instead, the "real names" always started with uppercase, moreover, the real ones usually are placed after words like "named", "is" etc...so it'is possible to clean this issue using a regex

# **Code**

# In[80]:


# extract correct names from text
df_archive2['name'] = df_archive2['text'].str.extract(r'([is|named|to|meet]+\s[A-Z][\W]?[A-Z]?[a-z]+)', expand = False).str.split(' ').str.get(1)


# In[81]:


# Change 'I' to NaN
df_archive2.loc[df_archive2['name'] == "I", 'name'] = np.nan


# **Test**

# In[82]:


df_archive2['name'].value_counts()


# In[83]:


df_archive2.head()


# In[84]:


df_archive2.info()


# ### Tidiness
# 1. Melt 'doggo' 'floofer' 'pupper' 'puppo' in `df_archive`.

# In[85]:


df_archive2.head(1)


# In[86]:


df_archive2['stage'] = df_archive2.doggo + df_archive2.floofer + df_archive2.pupper + df_archive2.puppo
df_archive2.head(1)


# In[87]:


df_archive2.stage.value_counts()


# In[88]:


df_archive2.loc[df_archive2.stage == 'NoneNoneNoneNone', 'stage'] = 'None'
df_archive2.loc[df_archive2.stage == 'NoneNonepupperNone', 'stage'] = 'pupper'
df_archive2.loc[df_archive2.stage == 'doggoNoneNoneNone', 'stage'] = 'doggo'
df_archive2.loc[df_archive2.stage == 'NoneNoneNonepuppo', 'stage'] = 'puppo'
df_archive2.loc[df_archive2.stage == 'doggoNonepupperNone', 'stage'] = 'multiple'
df_archive2.loc[df_archive2.stage == 'NoneflooferNoneNone', 'stage'] = 'floofer'
df_archive2.loc[df_archive2.stage == 'doggoflooferNoneNone', 'stage'] = 'multiple'
df_archive2.loc[df_archive2.stage == 'doggoNoneNonepuppo', 'stage'] = 'multiple'


# In[89]:


df_archive2.stage.value_counts()


# In[90]:


# drop individual stages columns
df_archive2.drop(['floofer','doggo','pupper', 'puppo'], axis=1, inplace = True)
df_archive2.head(1)


# In[91]:


df_pred.info()


# 2. Join all data frames into `twitter_archive_master.csv`.

# In[92]:


twitter_archive_master = df_archive2.merge(df_json, on=['tweet_id'], how='left')


# In[93]:


twitter_archive_master.info()


# In[94]:


twitter_archive_master.head(1)


# In[95]:


# save to csv
twitter_archive_master.to_csv('twitter_archive_master.csv', index=False)


# ## Analysis

# In[96]:


ax = twitter_archive_master.groupby('stage').favorite_count.mean().sort_values(ascending=False).plot(kind='bar', title='Favorite Count by stage of a dog');
ax.set_xlabel("Stage");
ax.set_ylabel("Favorite_count");


# In[97]:


ax = twitter_archive_master.groupby('stage').retweet_count.mean().sort_values(ascending=False).plot(kind='bar', title='Retweet Count by stage of a dog');
ax.set_xlabel("Stage");
ax.set_ylabel("Retweet_count");


# In[98]:


ax = twitter_archive_master.groupby('stage').rating_numerator.mean().sort_values(ascending=False).plot(kind='bar', title='Rating numerator by stage of a dog');
ax.set_xlabel("Stage");
ax.set_ylabel("rating_numerator");


# The chart shows that tweets that contain dogs' stages like 'puppo' or 'doggo' tend to have higher count favorite count. On the other hand, 'puppers' have the lowest favorite count, even lower then 'None'. The same patter is present in the 'retweet_count'. In addition, dogs that have 'stage' added to the tweets, tend to have higher rating numerators. 'Floofers' tend to have the greatest rating numerators.

# In[99]:


twitter_archive_master.rating_numerator.mean()


# Mean rating_numerator for the tweets is 10.84.

# In[100]:


twitter_archive_master[twitter_archive_master.stage == 'floofer'].rating_numerator.mean()


# Mean rating_numerator for floofers is 12.

# In[101]:


ax = twitter_archive_master.groupby('rating_numerator').favorite_count.mean().sort_values(ascending=False).plot(kind='bar', title='Favorite Count by rating numerator');
ax.set_xlabel("rating_numerator");
ax.set_ylabel("favorite_count");


# In[102]:


twitter_archive_master.rating_numerator.value_counts()


# The chart shows that dogs with higher ratings also tend to get higher favorite count. An exception is rating numerator of 0; however, only one dog deceived that rating; therefore, we can ignore that results as the sample size is not big enough.

# In[103]:


twitter_archive_master.favorite_count.mean()


# In[104]:


twitter_archive_master.groupby('rating_numerator').favorite_count.mean().sort_values(ascending=False)


# In[105]:


ax = twitter_archive_master.groupby('name').favorite_count.mean().sort_values(ascending=False).head().plot(kind='bar', title='Favorite Count by name');
ax.set_xlabel("Name");
ax.set_ylabel("Favorite Count");


# In[106]:


twitter_archive_master.groupby('name').favorite_count.mean().sort_values(ascending=False).head()


# In[107]:


ax = twitter_archive_master.groupby('name').favorite_count.mean().sort_values(ascending=False).tail().plot(kind='bar', title='Favorite Count by name');
ax.set_xlabel("Name");
ax.set_ylabel("Favorite Count");


# In[108]:


twitter_archive_master.groupby('name').favorite_count.mean().sort_values(ascending=False).tail()

