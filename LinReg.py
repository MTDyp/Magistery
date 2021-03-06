#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import os

cwd = os.getcwd()                       
os.chdir("C:\\Users\\MPC\\Downloads")              #Выбор папки, откуда берется файл
cwd


# In[89]:


table = pd.read_excel('Rus66.xls')
table.head(10)
table.info()

#  David = dataset.copy()   # Один раз подготовить таблицу и работать с копией
# In[90]:


table.drop([0,1], inplace=True)
# table.dropna(axis = 'columns', inplace = True)


# In[91]:


table.rename(columns={'Время': 'Time', 'Tвн': 'T_home, C', 'Tнв':'T_Inside'}, inplace=True)
table.head()


# In[92]:

dataset = dataset.query("~(Q <= 0)")              #Отсекаю значеня Q, которые равны или меньше 0 
table.describe()


# In[93]:


table.plot(x='T_Inside', y='Q', style='o')  
plt.title('Зависимость Q от наружной температуры')  
plt.xlabel('Наружная температура, C')  
plt.ylabel('Потребление энергии, Гкалл')  
plt.show()


# In[94]:


plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(table['Q'])


# In[95]:


X = table['T_Inside'].values.reshape(-1,1)
y = table['Q'].values.reshape(-1,1)


# In[53]:
a = []
for i in y:
    for j in i:
        a.append(j) 
seabornInstance.distplot(a)

print (a)


# In[117]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[118]:


regressor = LinearRegression()  
regressor.fit(X_train, y_train)
print(regressor.intercept_)#For retrieving the slope:
print(regressor.coef_)


# In[119]:


y_pred = regressor.predict(X_test)


# In[120]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[123]:


plt.scatter(X_test, y_test,  color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


#  Применение 3х метрик оценки регрессии
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


## Множественная линейная регрессия
dataset = pd.read_excel('Davidova28nT(12.19-05.20).xls') # Работаю с данными с Давыдова, т.к. там есть доступ к термометру в доме
dataset.rename(columns={'Время': 'Time', 'Tвн': 'T_home, C', 'Tнв':'T_Inside'}, inplace=True) # Переименование столбцов

dataset.drop([0,1], inplace=True)  # очистка ненужных столбцов с excel таблицы

dataset.isnull().any()      # проверка на нулевые столбцы

dataset.shape

dataset.describe()

X = dataset[['T1','M1', 'dP']].values
y = dataset['Q'].values

Q = []                           # Преобразование N-размерного массива в одномерный
for i in X:
    for j in i:
        Q.append(j) 
        
        

        
plt.figure(figsize =(15, 10))
plt.tight_layout()
seabornInstance.distplot(Q)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df

df1 = df.head(25)                                                           # визуализация обучения на 25 примерах в виде гистограммы
df1.plot(kind = 'bar', figsize=(10,8))
plt.grid(which='major', linestyle ='-', linewidth='0.5', color = 'green')
plt.grid(which='minor', linestyle =':', linewidth='0.5', color = 'black')
plt.show()


#  Применение 3х метрик оценки регрессии
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
