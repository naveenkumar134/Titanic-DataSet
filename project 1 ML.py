#!/usr/bin/env python
# coding: utf-8

# # Titanic DataSet.
# 
# 
# # Import Libraries ....
# 

# lets import some libraries to get started
# 

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # The Data....
#        let s start the dataset read the titanic train_csv in pandas dataframe

# In[5]:


train=pd.read_csv('titanic_train.csv')


# In[6]:


train.head()


# # Exploratory Data Analysis
#        let s begin some exploratory data analysis .i will checking misssing data
#        

# # Missing data
#       we can use seaborn to create simple heatmap to see where are missing data..

# In[9]:


train.isnull()


# In[11]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


#  Roughly 20 percent of age data missing the proposition of age missing is likely small enough for reasonble repalcement  form of   imputations lokking at the cabin columns..    

# In[16]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)


# In[21]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')


# In[26]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')


# In[32]:


sns.distplot(train['Age'].dropna(),kde=False,color='blue',bins=40)


# In[33]:


train['Age'].hist(bins=30,color='darkblue',alpha=0.3)


# In[35]:


sns.countplot(x='SibSp',data=train)


# In[36]:


train['Fare'].hist(color='green',bins=40,figsize=(8,4))


# # Data Cleaning
#     we want fill missing age data instead of just droping the missing age data rows. ones way to do this filling in mean age of all pessangers (imputation) check average age by passengers class ..

# In[38]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# In[69]:


def impute_age(cols):
    Age = cols[0]
    Pclass =cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        
        elif Pclass == 2:
            return 24
        
        else:
            return 24
    else:
        return Age


# In[70]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[71]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[73]:


train.drop('Cabin',axis=1,inplace=True)


# In[74]:


train.head()


# In[75]:


train.dropna(inplace=True)


# # Converting Categorical Features
#      we need to convert caterogical features using pandas otherwise our mechine learning algorithm wonte to be directly able to be more feature in inputs

# In[76]:


train.info()


# In[77]:


pd.get_dummies(train['Embarked'],drop_first=True).head()


# In[79]:


sex=pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)


# In[81]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[86]:


train.head()

trian =pd.concat([,train,sex,embark],axis=1)
# In[87]:


train.head()


# # Building Logistic Regression Model
#   lets start spilting our data into a training set and test set..

# # Train Test Spilt
# 

# In[88]:


train.drop('Survived',axis=1).head()


# In[89]:


train['Survived'].head()


# In[90]:


from sklearn.model_selection import train_test_split


# In[93]:


x_test,x_train,y_test,y_train= train_test_split(train.drop('Survived',axis=1),
                                               train['Survived'],test_size=0.30,
                                                random_state=101)


# # Training and Predicting

# In[96]:


from sklearn.linear_model import LogisticRegression 


# In[98]:


logmodel=LogisticRegression()
logmodel.fit(x_train,y_train)


# In[99]:


predictions=logmodel.predict(x_test)


# In[100]:


from sklearn.metrics import confusion_matrix


# In[101]:


accuracy=confusion_matrix(y_test,predictions)
accuracy


# In[104]:


from sklearn.metrics import accuracy_score


# In[106]:


accuracy=accuracy_score(y_test,predictions)
accuracy


# In[107]:


predictions


# In[85]:


train.head()


# In[ ]:




