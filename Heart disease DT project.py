#!/usr/bin/env python
# coding: utf-8

# ## import the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config
set_config(print_changed_only=False)


# ## import the data

# In[2]:


data= pd.read_csv('Heart disease.csv')
data.head()


# ## Eyeball the data

# In[3]:


df=data.copy()
df


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


# we see that data doesn't have null values


# In[7]:


df.describe()


# ## data preprocessing and cleaning

# In[8]:


df.isnull().sum()


# In[9]:


df=df.dropna(axis=0)
df.isnull().sum()


# In[10]:


df.shape


# In[11]:


## we see that education and other columns that are in float need to be changed to int for better understanding
df['education']=df['education'].astype(int)


# In[12]:


df.columns


# In[13]:


df['cigsPerDay'].unique()


# In[14]:


df['cigsPerDay']=df['cigsPerDay'].astype(int)


# In[15]:


df.head()


# In[16]:


df['BPMeds'].unique()


# In[17]:


df['BPMeds']=df['BPMeds'].astype(int)


# In[18]:


df.columns


# In[19]:


df['totChol'].unique()


# In[20]:


df['totChol']=df['totChol'].astype(int)


# In[21]:


df['sysBP'].unique()


# In[22]:


df['diaBP'].unique()


# In[23]:


df['heartRate'].unique()


# In[24]:


df['heartRate']=df['heartRate'].astype(int)


# In[25]:


df['glucose'].unique()


# In[26]:


df['glucose']=df['glucose'].astype(int)


# In[27]:


df.info()


# In[28]:


df.head()


# In[29]:


## now we have given proper datatypes to our features


# In[30]:


# check if dataset is balanced (what % of targets are 1s)
# targets.sum() will give us the number of 1s that there are
# the shape[0] will give us the length of the targets array
## here target is 'TenYearCHD'
df['TenYearCHD'].sum()/df.shape[0]


# In[31]:


## we see that the data is not balanced


# In[32]:


df['TenYearCHD'].sum()


# In[33]:


df.shape[0]


# In[34]:


##let's plot the countplot to visualize this abnormality better 
sns.countplot(data=df,x='TenYearCHD')


# In[35]:


# let's continue our process with this data set only ...later we will try it with balanced dataset as well


# # EDA

# In[36]:


df.columns


# In[ ]:





# In[ ]:





# In[37]:


## to check the correlation of variables with one another and with the target variable
plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True)


# In[38]:


# we see that our target feature is correlated with age, prevalentHyp, 'sysBP',diaBP, and glucose


# ## let's look at the distribution of these features

# In[39]:


sns.distplot(df['age'])


# In[40]:


sns.distplot(df['sysBP'])


# In[41]:


sns.distplot(df['diaBP'])


# In[42]:


sns.distplot(df['glucose'])


# ## Let's define our targets and inputs

# In[43]:


X=df.drop(['TenYearCHD'],axis=1)
Y=df['TenYearCHD']


# ## Create Train and Test datsets

# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=100)


# In[46]:


print(X_train.shape[0],X_test.shape[0])


# # import the model and fit on training dataset

# In[47]:


from sklearn.tree import DecisionTreeClassifier


# In[48]:


dtc=DecisionTreeClassifier()


# In[49]:


dtc.fit(X_train,Y_train)


# In[50]:


## let's predict  the values on test dataset
y_pred=dtc.predict(X_test)


# ## check the accuracy of the model

# In[51]:


from sklearn import metrics


# In[52]:


metrics.accuracy_score(Y_test,y_pred)


# In[53]:


from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(Y_test,y_pred)
print(confusion)


# In[54]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,y_pred))


# ## let's use K-fold validation

# In[55]:


from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
cv=RepeatedKFold(n_splits=5,n_repeats=3,random_state=10)


# In[56]:


scores=cross_val_score(dtc,X,Y,scoring='accuracy',cv=cv,n_jobs=-1)


# In[57]:


scores.mean()


# In[58]:


from sklearn import tree
plt.figure(figsize=(20,20))
tree.plot_tree(dtc)
("")


# ## let's run a grid search to tune the hyperparameters

# In[59]:


print(dtc.get_params)


# In[61]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':[2,3,4,5,6,7,8,9,None],
    'criterion':['gini','entropy'],
    
}

grid_search = GridSearchCV(estimator = dtc, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(X_train,Y_train)


# In[68]:


grid_search.best_params_


# In[297]:


dtc2=DecisionTreeClassifier(criterion = 'gini', max_depth = 2)


# In[298]:


dtc2.fit(X_train, Y_train)


# In[299]:


y_pred2=dtc2.predict(X_test)


# ## check the accuracy of the model

# In[300]:


metrics.accuracy_score(Y_test,y_pred2)


# In[301]:



confusion = confusion_matrix(Y_test,y_pred2)
print(confusion)


# In[302]:


print(classification_report(Y_test,y_pred2))


# In[303]:


scores=cross_val_score(dtc2,X,Y,scoring='accuracy',cv=cv,n_jobs=-1)


# In[304]:


scores.mean()


# In[305]:


## our confusion matrix shows that the model has become biased and it doesn't perform well with the Gridsearch 


# In[306]:


# we should try balancing the dataset


# In[307]:


from imblearn.over_sampling import SMOTE


# In[308]:


oversample=SMOTE()


# In[309]:


X,Y=oversample.fit_resample(X,Y)


# In[310]:


Y.sum()


# In[311]:


Y.shape


# In[312]:


Y.sum()/Y.shape[0]


# In[313]:


## now we see that the dataset is balanced


# ## now we re run all the commands to see if our model performed better

# In[316]:


X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=100)


# In[317]:


print(X_train.shape[0],X_test.shape[0])


# # fit model on training dataset

# In[319]:


dtc3=DecisionTreeClassifier()


# In[320]:


dtc3.fit(X_train,Y_train)


# In[321]:


## let's predict  the values on test dataset
y_pred3=dtc3.predict(X_test)


# ## check the accuracy of the model

# In[322]:


metrics.accuracy_score(Y_test,y_pred3)


# In[323]:


confusion = confusion_matrix(Y_test,y_pred3)
print(confusion)


# In[324]:


print(classification_report(Y_test,y_pred3))


# In[325]:


scores=cross_val_score(dtc,X,Y,scoring='accuracy',cv=cv,n_jobs=-1)


# In[326]:


scores.mean()


# In[377]:


## so we see our model's accuracy has slightly increased after balancing


# In[378]:


import pickle


# In[379]:


with open('model','wb') as file:
    pickle.dump(dtc3,file)


# In[ ]:




