
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("student_performance_2.csv")


# In[3]:


df.isnull()


# In[4]:


df.isnull().sum()


# In[5]:


df.isnull().sum().sum()


# In[6]:


df['Math Score'] = df['Math Score'].fillna(df['Math Score'].mean())


# In[7]:


df.isnull().sum()


# In[8]:


df['Reading Score'] = df['Reading Score'].fillna(df['Reading Score'].median())


# In[9]:


df.isnull().sum()


# In[10]:


df['Writing Score'] = df['Writing Score'].fillna(df['Writing Score'].min())


# In[11]:


df.isnull().sum()


# In[12]:


df.dtypes


# In[13]:


df.describe()


# In[14]:


df['Age'] = df['Age'].astype('float')


# In[15]:


df.dtypes


# In[16]:


df.describe()


# In[17]:


df['Age'] = df['Age'].astype('int')


# In[18]:


df.dtypes


# In[19]:


df.count()


# In[20]:


df['Age'].value_counts()


# In[29]:


df.describe()


# In[22]:


df['Class Performance'].value_counts()


# In[23]:


df.info()


# In[24]:


# data normalization
def min_max_normalize(
    name: str
):
    df[ name ] = (df[ name ] - df[ name ].min()) / ( df[ name ].max() - df[ name ].min() )


# In[25]:


min_max_normalize( "Age" )


# In[26]:


df.loc[ df.Gender == "Male" , "Gender" ] = 0
df.loc[ df.Gender == "Female" , "Gender" ] = 1


# In[27]:


df.describe()


# In[28]:


df


# In[30]:


dimensions = df.shape
print(dimensions)


# In[31]:


missing_values = df.isnull().sum()


# In[32]:


description = df.describe()


# In[33]:


df


# In[34]:


df.isnull().sum(0)


# In[35]:


df["Math Score"].unique()


# In[36]:


from sklearn import preprocessing


# In[37]:


label_encoder = preprocessing.LabelEncoder()


# In[38]:


df["Math Score"]= label_encoder.fit_transform(df["Math Score"])


# In[39]:


df["Math Score"].unique()


# In[40]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df["Age"]= label_encoder.fit_transform(df["Age"])
df["Age"].unique()


# In[42]:


import matplotlib.pylot as plt


# In[43]:


import matplotlib.pyplot as plt


# In[44]:


xpoints = np.array([1, 8])
ypoints = np.array([3, 10])

plt.plot(xpoints, ypoints)
plt.show()


# In[45]:


Age = [22,55,62,45,21,22,34,42,42,4,2,102,95,85,55,110,120,70,65,55,111,115,80,75,65,54,44,43,42,48]
Gender = [Male, Female]
plt.hist(Age, Gender, histtype='bar', rwidth=0.8)
plt.xlabel('age groups')
plt.ylabel('Gender')
plt.title('Histogram')
plt.show()


# In[46]:


Age = [22,55,62,45,21,22,34,42,42,4,2,102,95,85,55,110,120,70,65,55,111,115,80,75,65,54,44,43,42,48]
Gender = [0,1]
plt.hist(Age, Gender, histtype='bar', rwidth=0.8)
plt.xlabel('age groups')
plt.ylabel('Gender')
plt.title('Histogram')
plt.show()

