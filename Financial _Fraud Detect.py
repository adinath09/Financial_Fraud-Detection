#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imported the  libraries
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data set - ONLNE PAYMENT FRAUD DETECTION.CSV
Fraud_D = pd.read_csv(r'C:\Users\ugale\Downloads\onlinefraud.csv')


# In[3]:


# Rename the column header

Fraud_D.columns= ["step", "type", "amount", "customer_starting_transaction", "bal_before_transaction", 
            "bal_after_transaction", "recipient_of_transaction", "bal_of_recepient_before_transaction", "bal_of_receipient_after_transaction", "fraud_transaction", "is_flaggeed_Fraud"]


# In[4]:


Fraud_D.head()


# In[5]:


Fraud_D.tail()


# In[6]:


Fraud_D.info()


# In[7]:


Fraud_D.describe()


# In[8]:


Fraud_D.describe().astype(int)


# In[9]:


Fraud_D.isnull()


# In[10]:


Fraud_D.isnull().sum()


# In[11]:


plt.figure(figsize = (10,5))
plt.title ("missing data visualization in the dataset")
sns.heatmap(Fraud_D.isnull(), cbar =True, cmap= "Blues_r")


# In[12]:


Fraud_D.shape


# In[13]:


# Univariate Analysis
#visualize type of online transaction
plt.figure(figsize=(10,5))
sns.countplot (x="type", data= Fraud_D)
plt.title ("Visualizing type of online transaction")
plt.xlabel("Type of online transaction")
plt.ylabel("count of online transaction type ")


# In[14]:


# create a function that properly labels isFraud

def Fraud (x):
    if x ==1:
        return "Fraudulent"
    else:
        return "not Fraudulent"
    
# create a new column
Fraud_D["fraud_transaction_label"] = Fraud_D["fraud_transaction"].apply(Fraud)


# create visualization
plt.figure(figsize = (10,5))
plt.title ("Fraudulent Transactions")
Fraud_D.fraud_transaction_label.value_counts().plot.pie(autopct='%1.1f%%')


# In[15]:


Fraud_D.fraud_transaction_label.value_counts()


# In[16]:


1142/1047433*100


# In[17]:


#To disable warnings
import warnings
warnings.filterwarnings("ignore")


plt.figure(figsize=(15,6))
sns.distplot(Fraud_D['step'],bins=100)


# In[18]:


# Visualization for amount column

sns.histplot(x= "amount", data =Fraud_D)


# In[19]:


Fraud_D.head()


# In[20]:


Fraud_D.tail()


# In[21]:


# Bivariate Analysis

sns.barplot(x='type',y='amount',data=Fraud_D)


# In[22]:


sns.jointplot(x='step',y='amount',data=Fraud_D)


# In[23]:


sns.scatterplot(x=Fraud_D["amount"], y=Fraud_D["step"])


# In[24]:


plt.figure(figsize=(15,6))
plt.scatter(x='amount',y='fraud_transaction_label',data=Fraud_D)
plt.xlabel('amount')
plt.ylabel('fraud_transaction_label')


# In[25]:


plt.scatter(x='type',y='fraud_transaction_label',data=Fraud_D)
plt.xlabel('type')
plt.ylabel('fraud_transaction_label')


# In[26]:


plt.figure(figsize=(12,8))
sns.countplot(x='fraud_transaction_label',data=Fraud_D,hue='type')
plt.legend(loc=[0.85,0.8])


# In[27]:


sns.boxplot(x= "type", y= "step", hue ="fraud_transaction_label", data= Fraud_D)


# In[28]:


sns.pairplot(Fraud_D)


# In[49]:


# Correlation

corel= Fraud_D.corr()
sns.heatmap(corel, annot =False)


# In[30]:


# One Hot Encoding
#1. select categorical variables

categorical = ['type']


# In[31]:


#2. use pd.get_dummies() for one hot encoding
#replace pass with your code

categories_dummies = pd.get_dummies(Fraud_D[categorical])

#view what you have done
categories_dummies.head()


# In[32]:


#join the encoded variables back to the main dataframe using pd.concat()
#pass both data and categories_dummies as a list of their names
#pop out documentation for pd.concat() to clarify

Fraud_D = pd.concat([Fraud_D,categories_dummies], axis=1)

#check what you have done
print(Fraud_D.shape)
Fraud_D.head()


# In[33]:


#remove the initial categorical columns now that we have encoded them
#use the list called categorical to delete all the initially selected columns at once

Fraud_D.drop(categorical, axis = 1, inplace = True)

Fraud_D.drop(columns=['fraud_transaction_label', 'customer_starting_transaction', 'recipient_of_transaction'], inplace=True)


# In[34]:


Fraud_D.head()


# In[35]:


y = Fraud_D.fraud_transaction


# In[36]:


X = Fraud_D.drop(['fraud_transaction'], axis = 1)


# In[37]:


X


# In[38]:


#import the libraries we will need
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# In[39]:


## Train test split( training on 80% while testing is 20%)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[40]:


# Initialize each models
LR = LogisticRegression(random_state=42)
KN = KNeighborsClassifier()
DC = DecisionTreeClassifier(random_state=42)
RF = RandomForestClassifier(random_state=42)


# In[41]:


#create list of your model names
models = [LR,KN,DC,RF]


# In[42]:


def plot_confusion_matrix(y_test,prediction):
    cm_ = confusion_matrix(y_test,prediction)
    plt.figure(figsize = (6,4))
    sns.heatmap(cm_, cmap ='coolwarm', linecolor = 'white', linewidths = 1, annot = True, fmt = 'd')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


# In[43]:


from sklearn.metrics import confusion_matrix


# In[44]:


#create function to train a model and evaluate accuracy
def trainer(model,X_train,y_train,X_test,y_test):
    #fit your model
    model.fit(X_train,y_train)
    #predict on the fitted model
    prediction = model.predict(X_test)
    #print evaluation metric
    print('\nFor {}, Accuracy score is {} \n'.format(model.__class__.__name__,accuracy_score(prediction,y_test)))
    print(classification_report(y_test, prediction)) #use this later
    plot_confusion_matrix(y_test,prediction)


# In[45]:


#loop through each model, training in the process
for model in models:
    trainer(model,X_train,y_train,X_test,y_test)
    


# In[48]:


# Importing the library to perform cross-validation
from sklearn.model_selection import cross_validate

# Running the cross-validation on both Decision Tree and Random Forest models; specifying recall as the scoring metric
DC_scores = cross_validate(DC, X_test, y_test, scoring='recall_macro')
RF_scores = cross_validate(RF, X_test, y_test, scoring='recall_macro')

# Printing the means of the cross-validations for both models
print('Decision Tree Recall Cross-Validation:', np.mean(DC_scores['test_score']))
print('Random Forest Recall Cross-Validation:', np.mean(RF_scores['test_score']))


# In[ ]:




