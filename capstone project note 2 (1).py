#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from tabulate import tabulate


# In[6]:


pip install tabulate


# In[7]:


df=pd.read_csv('HR.csv')
df.head(10)


# In[8]:


df.info()


# In[9]:


pd.options.display.max_columns = None


# In[10]:


df.head()


# In[11]:


df=df.drop(['IDX','Applicant_ID','Department','Organization','Passing_Year_Of_Graduation','Passing_Year_Of_PG','University_Grad','University_PG','University_PHD','Passing_Year_Of_PHD','Curent_Location','No_Of_Companies_worked'],axis=1)
df.head()


# In[115]:


df.shape


# In[116]:


df.info()


# In[117]:


dups = df.duplicated()
print('Number of duplicate rows = %d' % (dups.sum()))
df[dups]


# In[15]:


df.describe().T


# In[16]:


df.isnull().sum()


# In[17]:


plt.figure(figsize=(7,6))
sns.countplot(x='Inhand_Offer', data=df)


# In[18]:


plt.figure(figsize=(7,6))
sns.countplot(x='Number_of_Publications', data=df)


# In[19]:


plt.figure(figsize=(10,6))
df.groupby('Certifications').sum()['Expected_CTC'].sort_values(ascending = True).plot(kind = 'bar')


# In[20]:


plt.figure(figsize=(10,4))
sns.countplot(x=df['Preferred_location'])


# In[21]:


pd.crosstab(df.Education, df.Inhand_Offer, margins=True,normalize=True)


# In[22]:


sns.scatterplot(x='Expected_CTC', y='Current_CTC', hue='Total_Experience', data=df)


# In[23]:


plt.figure(figsize=(10,4))
sns.scatterplot(x='Expected_CTC', y='Current_CTC', hue='Total_Experience_in_field_applied', data=df)


# In[24]:


sns.barplot(x='Education', y='Expected_CTC', data=df)


# In[25]:


plt.figure(figsize=(20,4))
sns.barplot(x='Preferred_location', y='Expected_CTC', data=df)


# In[26]:


plt.figure(figsize=(7,4))
sns.barplot(x='Certifications', y='Expected_CTC', data=df)


# In[27]:


sns.barplot(x='Inhand_Offer', y='Expected_CTC', data=df)


# In[28]:


plt.figure(figsize=(25,6))
sns.barplot(x='Designation', y='Expected_CTC', data=df)


# In[29]:


plt.figure(figsize=(10,4))
sns.barplot(x='Number_of_Publications', y='Expected_CTC', data=df)


# In[30]:


df.Designation.value_counts()


# In[31]:


df.Industry.value_counts()


# In[32]:


df.Role.value_counts()


# In[33]:


df.Graduation_Specialization.value_counts()


# In[34]:


df['Graduation_Specialization']=np.where(df['Graduation_Specialization']=='Chemistry','1',df['Graduation_Specialization'])
df['Graduation_Specialization']=np.where(df['Graduation_Specialization']=='Economics','2',df['Graduation_Specialization'])
df['Graduation_Specialization']=np.where(df['Graduation_Specialization']=='Mathematics','3',df['Graduation_Specialization'])
df['Graduation_Specialization']=np.where(df['Graduation_Specialization']=='Zoology','4',df['Graduation_Specialization'])
df['Graduation_Specialization']=np.where(df['Graduation_Specialization']=='Arts','5',df['Graduation_Specialization'])
df['Graduation_Specialization']=np.where(df['Graduation_Specialization']=='Psychology','6',df['Graduation_Specialization'])
df['Graduation_Specialization']=np.where(df['Graduation_Specialization']=='Sociology','7',df['Graduation_Specialization'])
df['Graduation_Specialization']=np.where(df['Graduation_Specialization']=='Botony','8',df['Graduation_Specialization'])
df['Graduation_Specialization']=np.where(df['Graduation_Specialization']=='Engineering','9',df['Graduation_Specialization']) 
df['Graduation_Specialization']=np.where(df['Graduation_Specialization']=='Others','1',df['Graduation_Specialization'])
df['Graduation_Specialization']=np.where(df['Graduation_Specialization']=='Statistics','10',df['Graduation_Specialization'])                                        
df['Graduation_Specialization']=np.where(df['Graduation_Specialization']=='Others','11',df['Graduation_Specialization'])


# In[35]:


df.PG_Specialization.value_counts()


# In[36]:


df['PG_Specialization']=np.where(df['PG_Specialization']=='Mathematics','1',df['PG_Specialization'])
df['PG_Specialization']=np.where(df['PG_Specialization']=='Chemistry','2',df['PG_Specialization'])
df['PG_Specialization']=np.where(df['PG_Specialization']=='Economics','3',df['PG_Specialization'])
df['PG_Specialization']=np.where(df['PG_Specialization']=='Engineering','4',df['PG_Specialization'])
df['PG_Specialization']=np.where(df['PG_Specialization']=='Statistics','5',df['PG_Specialization'])
df['PG_Specialization']=np.where(df['PG_Specialization']=='Others','6',df['PG_Specialization'])
df['PG_Specialization']=np.where(df['PG_Specialization']=='Psychology','7',df['PG_Specialization'])
df['PG_Specialization']=np.where(df['PG_Specialization']=='Zoology','8',df['PG_Specialization'])
df['PG_Specialization']=np.where(df['PG_Specialization']=='Arts','9',df['PG_Specialization'])
df['PG_Specialization']=np.where(df['PG_Specialization']=='Sociology','10',df['PG_Specialization'])
df['PG_Specialization']=np.where(df['PG_Specialization']=='Botony','11',df['PG_Specialization'])


# In[37]:


df.PHD_Specialization.value_counts()


# In[38]:


df['PHD_Specialization']=np.where(df['PHD_Specialization']=='Others','1',df['PHD_Specialization'])
df['PHD_Specialization']=np.where(df['PHD_Specialization']=='Chemistry','2',df['PHD_Specialization'])
df['PHD_Specialization']=np.where(df['PHD_Specialization']=='Mathematics','3',df['PHD_Specialization'])
df['PHD_Specialization']=np.where(df['PHD_Specialization']=='Economics','4',df['PHD_Specialization'])
df['PHD_Specialization']=np.where(df['PHD_Specialization']=='Engineering','5',df['PHD_Specialization'])
df['PHD_Specialization']=np.where(df['PHD_Specialization']=='Statistics','6',df['PHD_Specialization'])
df['PHD_Specialization']=np.where(df['PHD_Specialization']=='Zoology','7',df['PHD_Specialization'])
df['PHD_Specialization']=np.where(df['PHD_Specialization']=='Sociology ','8',df['PHD_Specialization'])
df['PHD_Specialization']=np.where(df['PHD_Specialization']=='Psychology','9',df['PHD_Specialization'])
df['PHD_Specialization']=np.where(df['PHD_Specialization']=='Botony','10',df['PHD_Specialization'])
df['PHD_Specialization']=np.where(df['PHD_Specialization']=='Arts','11',df['PHD_Specialization'])


# In[39]:


df['Role']=np.where(df['Role']=='Others','1',df['Role'])
df['Role']=np.where(df['Role']=='Bio statistician','2',df['Role'])
df['Role']=np.where(df['Role']=='Analyst','3',df['Role'])
df['Role']=np.where(df['Role']=='Project Manager','4',df['Role'])
df['Role']=np.where(df['Role']=='Team Lead','5',df['Role'])
df['Role']=np.where(df['Role']=='Consultant','6',df['Role'])
df['Role']=np.where(df['Role']=='Business Analyst','7',df['Role'])
df['Role']=np.where(df['Role']=='Sales Execituve','8',df['Role'])
df['Role']=np.where(df['Role']=='Sales Manager','9',df['Role'])
df['Role']=np.where(df['Role']=='Senior Researcher ','10',df['Role'])
df['Role']=np.where(df['Role']=='Financial Analyst','11',df['Role'])
df['Role']=np.where(df['Role']=='CEO','12',df['Role'])
df['Role']=np.where(df['Role']=='Scientist','13',df['Role'])
df['Role']=np.where(df['Role']=='Head','14',df['Role'])
df['Role']=np.where(df['Role']=='Associate','15',df['Role'])
df['Role']=np.where(df['Role']=='Data scientist','16',df['Role'])
df['Role']=np.where(df['Role']=='Principal Analyst','17',df['Role'])
df['Role']=np.where(df['Role']=='Area Sales Manager','18',df['Role'])
df['Role']=np.where(df['Role']=='Senior Analyst','19',df['Role'])
df['Role']=np.where(df['Role']=='Researcher','20',df['Role'])
df['Role']=np.where(df['Role']=='Sr. Business Analyst','21',df['Role'])
df['Role']=np.where(df['Role']=='Professor','22',df['Role'])
df['Role']=np.where(df['Role']=='Research Scientist','23',df['Role'])
df['Role']=np.where(df['Role']=='Lab Executuve','24',df['Role'])


# In[40]:


df['Designation']=np.where(df['Designation']=='HR','1',df['Designation'])
df['Designation']=np.where(df['Designation']=='Others','2',df['Designation'])
df['Designation']=np.where(df['Designation']=='Manager','3',df['Designation'])
df['Designation']=np.where(df['Designation']=='Product Manager','4',df['Designation'])
df['Designation']=np.where(df['Designation']=='Sr.Manager','5',df['Designation'])
df['Designation']=np.where(df['Designation']=='Consultant','6',df['Designation'])
df['Designation']=np.where(df['Designation']=='Marketing Manager','7',df['Designation'])
df['Designation']=np.where(df['Designation']=='Assistant Manager','8',df['Designation'])
df['Designation']=np.where(df['Designation']=='Data Analyst','9',df['Designation'])
df['Designation']=np.where(df['Designation']=='Research Analyst','10',df['Designation'])
df['Designation']=np.where(df['Designation']=='Medical Officer','11',df['Designation'])
df['Designation']=np.where(df['Designation']=='Software Developer','12',df['Designation'])
df['Designation']=np.where(df['Designation']=='Web Designer','13',df['Designation'])
df['Designation']=np.where(df['Designation']=='Network Engineer','14',df['Designation'])
df['Designation']=np.where(df['Designation']=='Director','15',df['Designation'])
df['Designation']=np.where(df['Designation']=='CA','16',df['Designation'])
df['Designation']=np.where(df['Designation']=='Research Scientist','17',df['Designation'])
df['Designation']=np.where(df['Designation']=='Scientist','18',df['Designation'])


# In[41]:


df['Industry']=np.where(df['Industry']=='Training','1',df['Industry'])
df['Industry']=np.where(df['Industry']=='IT ','2',df['Industry'])
df['Industry']=np.where(df['Industry']=='Insurance','3',df['Industry'])
df['Industry']=np.where(df['Industry']=='BFSI','4',df['Industry'])
df['Industry']=np.where(df['Industry']=='Automobile','5',df['Industry'])
df['Industry']=np.where(df['Industry']=='Analytics','6',df['Industry'])
df['Industry']=np.where(df['Industry']=='Retail','7',df['Industry'])
df['Industry']=np.where(df['Industry']=='Telecom ','8',df['Industry'])
df['Industry']=np.where(df['Industry']=='Aviation ','9',df['Industry'])
df['Industry']=np.where(df['Industry']=='FMCG','10',df['Industry'])
df['Industry']=np.where(df['Industry']=='Others','11',df['Industry'])


# In[42]:


for feature in df.columns: 
    if df[feature].dtype == 'object': 
        print('\n')
        print('feature:',feature)
        print(pd.Categorical(df[feature].unique()))
        print(pd.Categorical(df[feature].unique()).codes)
        df[feature] = pd.Categorical(df[feature]).codes


# In[43]:


df.head()


# In[44]:


df.dtypes


# In[45]:


imputer = KNNImputer(n_neighbors=2)
imputer.fit_transform([df.Designation,df.Industry])


# In[46]:


df_na = df.isna().sum()
df_na[df_na.values > 0].sort_values(ascending=False)


# In[47]:


df.info()


# In[48]:


df.describe(include="all")["Current_CTC"]


# In[49]:


df.Current_CTC.unique()


# In[50]:


fig,axes = plt.subplots(nrows=4,ncols=4)
fig.set_size_inches(18,12)
sns.histplot(df['Total_Experience'], kde=True, ax=axes[0][0])
sns.histplot(df['Total_Experience_in_field_applied'], kde=True, ax=axes[0][1])
sns.histplot(df['Industry'], kde=True, ax=axes[0][2])
sns.histplot(df['Designation'], kde=True, ax=axes[0][3])
sns.histplot(df['Education'], kde=True, ax=axes[1][0])
sns.histplot(df['Preferred_location'], kde=True, ax=axes[1][1])
sns.histplot(df['Current_CTC'], kde=True, ax=axes[1][2])
sns.histplot(df['Inhand_Offer'], kde=True, ax=axes[1][3])
sns.histplot(df['Number_of_Publications'], kde=True, ax=axes[2][0])
sns.histplot(df['Certifications'], kde=True, ax=axes[2][1])
sns.histplot(df['International_degree_any'], kde=True, ax=axes[2][2])
sns.histplot(df['Expected_CTC'], kde=True, ax=axes[2][3])
sns.histplot(df['Role'], kde=True, ax=axes[3][0])
sns.histplot(df['Graduation_Specialization'], kde=True, ax=axes[3][1])
sns.histplot(df['PG_Specialization'], kde=True, ax=axes[3][2])
sns.histplot(df['PHD_Specialization'], kde=True, ax=axes[3][3])
plt.show()


# In[51]:


fig,axes = plt.subplots(nrows=4,ncols=4)
fig.set_size_inches(16,12)
sns.boxplot(x='Total_Experience', data=df, ax=axes[0][0])
sns.boxplot(x='Total_Experience_in_field_applied', data=df, ax=axes[0][1])
sns.boxplot(x='Industry', data=df, ax=axes[0][2])
sns.boxplot(x='Designation', data=df, ax=axes[0][3])
sns.boxplot(x='Education', data=df, ax=axes[1][0])
sns.boxplot(x='Preferred_location', data=df, ax=axes[1][1])
sns.boxplot(x='Current_CTC', data=df, ax=axes[1][2])
sns.boxplot(x='Inhand_Offer', data=df, ax=axes[1][3])
sns.boxplot(x='Number_of_Publications', data=df, ax=axes[2][0])
sns.boxplot(x='Certifications', data=df, ax=axes[2][1])
sns.boxplot(x='International_degree_any', data=df, ax=axes[2][2])
sns.boxplot(x='Expected_CTC', data=df, ax=axes[2][3])
sns.boxplot(x='Role', data=df, ax=axes[3][0])
sns.boxplot(x='Graduation_Specialization', data=df, ax=axes[3][1])
sns.boxplot(x='PG_Specialization', data=df, ax=axes[3][2])
sns.boxplot(x='PHD_Specialization', data=df, ax=axes[3][3])
plt.show()


# In[52]:


df.skew().sort_values(ascending=False)


# In[53]:


fig_size=(6,5)
sns.pairplot(df,diag_kind='kde')
plt.show()


# In[54]:


df.cov()


# In[55]:


plt.figure(figsize=(12,7))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Blues')
plt.show()


# In[56]:


plt.figure(figsize=(10,10))
df.boxplot(vert=0)
plt.title('Data with outliers',fontsize=16)
plt.show()


# In[57]:


fig_dims = (10, 5) 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims) 
sns.histplot((df['Role']), kde=True, ax = axs[0])
sns.boxplot(x= (df['Role']), ax = axs[1])
plt.title('Role Before outlier treatment')
plt.show()


# In[58]:


fig_dims = (10, 5) 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims) 
sns.histplot((df['International_degree_any']), kde=True, ax = axs[0])
sns.boxplot(x= (df['International_degree_any']), ax = axs[1])
plt.title('International_degree_any Before outlier treatment')
plt.show()


# In[59]:


fig_dims = (10, 5) 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims) 
sns.histplot(np.log(df['International_degree_any']), kde=True, ax = axs[0])
sns.boxplot(x= np.log(df['International_degree_any']), ax = axs[1])
plt.title('International_degree_any After outlier treatment')
plt.show()


# In[60]:


fig_dims = (10, 5) 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims) 
sns.histplot((df['Certifications']), kde=True, ax = axs[0])
sns.boxplot(x= (df['Certifications']), ax = axs[1])
plt.title('Certifications Before outlier treatment')
plt.show()


# In[61]:


fig_dims = (10, 5) 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims) 
sns.histplot(np.log(df['Certifications']), kde=True, ax = axs[0])
sns.boxplot(x= np.log(df['Certifications']), ax = axs[1])
plt.title('Certifications After outlier treatment')
plt.show()


# In[62]:


fig_dims = (10, 5) 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims) 
sns.histplot(np.log(df['Role']), kde=True, ax = axs[0])
sns.boxplot(x= np.log(df['Role']), ax = axs[1])
plt.title('Role CTC After outlier treatment')
plt.show()


# In[63]:


fig_dims = (10, 5) 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims) 
sns.histplot((df['Expected_CTC']), kde=True, ax = axs[0])
sns.boxplot(x= (df['Expected_CTC']), ax = axs[1])
plt.title('Expected CTC Before outlier treatment')
plt.show()


# In[64]:


fig_dims = (10, 5) 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims) 
sns.histplot(np.log(df['Expected_CTC']), kde=True, ax = axs[0])
sns.boxplot(x= np.log(df['Expected_CTC']), ax = axs[1])
plt.title('Expected CTC After outlier treatment')
plt.show()


# In[65]:


fig_dims = (10, 5) 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims) 
sns.histplot((df['Current_CTC']), kde=True, ax = axs[0])
sns.boxplot(x= (df['Current_CTC']), ax = axs[1])
plt.title('Current CTC Before outlier treatment')
plt.show()


# In[66]:


fig_dims = (10, 5) 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=fig_dims) 
sns.histplot(np.log(df['Current_CTC']), kde=True, ax = axs[0])
sns.boxplot(x= np.log(df['Current_CTC']), ax = axs[1])
print(df['Current_CTC'].skew())
plt.title('Current CTC After outlier treatment')
plt.show()


# In[67]:


fig,axes = plt.subplots(nrows=4,ncols=3)
fig.set_size_inches(16,12)
sns.distplot(df['Total_Experience'],kde = True, ax=axes[0][0])
sns.distplot(df['Total_Experience_in_field_applied'], kde = True, ax=axes[0][1])
sns.distplot(df['Industry'],  kde = True, ax=axes[0][2])
sns.distplot(df['Designation'],  kde = True, ax=axes[1][0])
sns.distplot(df['Education'],  kde = True, ax=axes[1][1])
sns.distplot(df['Preferred_location'],  kde = True, ax=axes[1][2])
sns.distplot(df['Current_CTC'],  kde = True, ax=axes[2][0])
sns.distplot(df['Inhand_Offer'],  kde = True, ax=axes[2][1])
sns.distplot(df['Number_of_Publications'],  kde = True, ax=axes[2][2])
sns.distplot(df['International_degree_any'],  kde = True, ax=axes[3][0])
sns.distplot(df['Certifications'],  kde = True, ax=axes[3][1])
sns.distplot(df['Expected_CTC'],  kde = True, ax=axes[3][2])
plt.show()


# In[68]:


plt.figure(figsize=(10,10))
df.boxplot(vert=0)
plt.title('Data with outliers',fontsize=16)
plt.show()


# In[69]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[70]:


# Copy all the predictor variables into X dataframe
X = df.drop('Expected_CTC', axis=1)

# Copy target into the y dataframe. 
y = df[['Expected_CTC']]


# In[71]:


X.head()


# In[72]:


y.head()


# In[73]:


# Split X and y into training and test set in 70:30 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , random_state=1)


# In[74]:


regression_model = LinearRegression()
regression_model.fit(X_train, y_train)


# In[75]:



for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))


# In[76]:


intercept = regression_model.intercept_[0]

print("The intercept for our model is {}".format(intercept))


# In[77]:


# R square on training data
regression_model.score(X_train, y_train)


# In[78]:


# R square on testing data
regression_model.score(X_test, y_test)


# In[79]:


predicted_train=regression_model.fit(X_train, y_train).predict(X_train)
np.sqrt(metrics.mean_squared_error(y_train,predicted_train))


# In[207]:


#RMSE on Testing data
predicted_test=regression_model.fit(X_train, y_train).predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test,predicted_test))


# In[244]:


x_pred=pd.DataFrame(regression_model.predict(X_test))
x_pred


# In[225]:


plt.scatter(y_test['Expected_CTC'], x_pred)


# In[226]:


print("MAE train:", metrics.mean_absolute_error(y_train, predicted_train))
print('MSEtrain:', metrics.mean_squared_error(y_train, predicted_train))
print('RMSEtrain:', np.sqrt(metrics.mean_squared_error(y_train,predicted_train)))
print("MAE test:", metrics.mean_absolute_error(y_test, predicted_test))
print('MSE test:', metrics.mean_squared_error(y_test, predicted_test))
print('RMSE test:', np.sqrt(metrics.mean_squared_error(y_test,predicted_test)))


# In[227]:


print("MAE test:", metrics.mean_absolute_error(y_test, predicted_test))
print('MSE test:', metrics.mean_squared_error(y_test, predicted_test))
print('RMSE test:', np.sqrt(metrics.mean_squared_error(y_test,predicted_test)))


# In[228]:


#RMSE on Training data
predicted_train=regression_model.fit(X_train, y_train).predict(X_train)
np.sqrt(metrics.mean_squared_error(y_train,predicted_train))


# In[229]:


#RMSE on Testing data
predicted_test=regression_model.fit(X_train, y_train).predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test,predicted_test))


# In[230]:


# concatenate X and y into a single dataframe
data_train = pd.concat([X_train, y_train], axis=1)
data_test=pd.concat([X_test,y_test],axis=1)
data_train.head()


# In[231]:


import statsmodels.formula.api as smf
lm1 = smf.ols(formula= 'Expected_CTC ~ Total_Experience+Total_Experience_in_field_applied+Industry+Designation+	Education+Preferred_location+Current_CTC+Inhand_Offer+Last_Appraisal_Rating+Number_of_Publications+Certifications+International_degree_any+Role+Graduation_Specialization+PG_Specialization+PHD_Specialization', data =  data_train).fit()
lm1.params


# In[232]:


print(lm1.summary())


# In[233]:


mse = np.mean((regression_model.predict(X_test)-y_test)**2)


# In[234]:


import math

math.sqrt(mse)


# In[235]:


regression_model.score(X_test, y_test)


# In[236]:


y_pred = regression_model.predict(X_test)


# In[237]:


plt.scatter(y_test['Expected_CTC'], y_pred)


# In[238]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[239]:


vif = [variance_inflation_factor(X.values, ix) for ix in range(X.shape[1])] 


# In[240]:


i=0
for column in X.columns:
    if i < 16:
        print (column ,"--->",  vif[i])
        i = i+1


# In[128]:


lm2 = smf.ols(formula= 'Expected_CTC ~ Total_Experience_in_field_applied+Industry+Designation+	Education+Preferred_location+Current_CTC+Inhand_Offer+Last_Appraisal_Rating+Number_of_Publications+Certifications+International_degree_any+Role+Graduation_Specialization+PG_Specialization+PHD_Specialization', data = data_train).fit()
lm2.params


# In[129]:


print(lm2.summary())


# In[130]:


mse = np.mean((regression_model.predict(X_test)-y_test)**2)
mse


# In[131]:


import math

math.sqrt(mse)


# In[132]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV


# In[133]:


x=df.drop(['Expected_CTC'],axis=1) #name is splitted in Model and Brand; Year is transformed to CarAge
y=df.Expected_CTC


# In[134]:


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=123,test_size=0.30)


# In[97]:


ss=StandardScaler() 
# we are scaling the data for ANN. Without scaling it will give very poor results. Computations becomes easier
x_train_scaled=ss.fit_transform(x_train)
x_test_scaled=ss.transform(x_test)


# In[123]:


annr = MLPRegressor(hidden_layer_sizes=(100),random_state=123, max_iter=2500)#you are free to tweak the layer sizes
rfr = RandomForestRegressor(random_state=123)
dtr = tree.DecisionTreeRegressor(random_state=123)
regression_model = LinearRegression()

models=[regression_model,dtr,rfr,annr]

rmse_train=[]
rmse_test=[]
scores_train=[]
scores_test=[]

for i in models:  # we are scaling the data for ANN. Without scaling it will give very poor results. Computations becomes easier
    
    if (i != annr) :
        i.fit(x_train,y_train)
        scores_train.append(i.score(x_train, y_train))
        scores_test.append(i.score(x_test, y_test))
        rmse_train.append(np.sqrt(mean_squared_error(y_train,i.predict(x_train))))
        rmse_test.append(np.sqrt(mean_squared_error(y_test,i.predict(x_test))))
 
    else :
        i.fit(x_train_scaled,y_train)
        scores_train.append(i.score(x_train_scaled, y_train))
        scores_test.append(i.score(x_test_scaled, y_test))
        rmse_train.append(np.sqrt(mean_squared_error(y_train,i.predict(x_train_scaled))))
        rmse_test.append(np.sqrt(mean_squared_error(y_test,i.predict(x_test_scaled))))
        
print(pd.DataFrame({'Train RMSE': rmse_train,'Test RMSE': rmse_test,'Training Score':scores_train,'Test Score': scores_test},
            index=['Linear Regression','Decision Tree Regressor','Random Forest Regressor', 'ANN Regressor']))


# In[135]:


from sklearn.tree import DecisionTreeRegressor


# In[136]:


dTree = DecisionTreeRegressor(random_state=1)
dTree.fit(X_train, y_train)


# In[137]:


print('DECISION TREE REGRESSOR MODEL TRAIN SCORE :',dTree.score(X_train, y_train))
print('DECISION TREE REGRESSOR MODEL TEST SCORE :',dTree.score(X_test, y_test))
print('RMSE test DECISION TREE train :',np.sqrt(metrics.mean_squared_error(y_train,dTree.predict(X_train))))
print('RMSE test DECISION TREE test:',np.sqrt(metrics.mean_squared_error(y_test,dTree.predict(X_test))))


# In[138]:


from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
cart = DecisionTreeRegressor()
Bagging_model=BaggingRegressor(base_estimator=cart,n_estimators=100,random_state=1)
Bagging_model.fit(X_train, y_train)


# In[139]:


y_train_predict = Bagging_model.predict(X_train)


# In[140]:


print('BAGGING MODEL TRAIN SCORE :', Bagging_model.score(X_train, y_train))
print('BAGGING MODEL TEST SCORE :',Bagging_model.score(X_test, y_test))


# In[141]:


print('RMSE test bagging:',np.sqrt(metrics.mean_squared_error(y_train,Bagging_model.predict(X_train))))


# In[142]:


print('BAGGING MODEL TRAIN SCORE :', Bagging_model.score(X_train, y_train))
print('BAGGING MODEL TEST SCORE :',Bagging_model.score(X_test, y_test))
print('RMSE train bagging:',np.sqrt(metrics.mean_squared_error(y_train,Bagging_model.predict(X_train))))
print('RMSE test bagging:',np.sqrt(metrics.mean_squared_error(y_test,Bagging_model.predict(X_test))))


# In[143]:


from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor


# In[144]:


abc = AdaBoostRegressor(random_state=1)
abc.fit(X_train,y_train)


# In[145]:


y_train_predict = abc.predict(X_train)


# In[146]:


print('ADA boosting MODEL TRAIN SCORE :',abc.score(X_train, y_train))
print('ADA boosting MODEL Test SCORE :',abc.score(X_test, y_test))
print('RMSE ADA train:',np.sqrt(metrics.mean_squared_error(y_train,abc.predict(X_train))))
print('RMSE ADA test:',np.sqrt(metrics.mean_squared_error(y_test,abc.predict(X_test))))


# In[147]:


from sklearn import datasets, ensemble


# In[148]:


gbc = GradientBoostingRegressor(random_state=1)
gbc.fit(X_train,y_train)


# In[149]:


y_train_predict = gbc.predict(X_train)


# In[150]:


print('Gredient boosting MODEL TRAIN SCORE :',gbc.score(X_train, y_train))
print('Gredient boosting MODEL Test SCORE :',gbc.score(X_test, y_test))
print('RMSE GBC train :',np.sqrt(metrics.mean_squared_error(y_train,gbc.predict(X_train))))
print('RMSE GBC test :',np.sqrt(metrics.mean_squared_error(y_test,gbc.predict(X_test))))


# In[ ]:




