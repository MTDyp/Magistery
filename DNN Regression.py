#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import explained_variance_score,     mean_absolute_error,     median_absolute_error
from sklearn.model_selection import train_test_split
import os


# In[2]:


cwd = os.getcwd()
os.chdir("C:\\Users\\MPC\\Downloads")
cwd
dataset = pd.read_excel('Davidova28nT(12.19-05.20).xls')

dataset.describe().T


# In[3]:


dataset.drop([0,1], inplace=True)
dataset.rename(columns={'Время': 'Time', 'Tвн': 'T_home', 'Tнв':'T_Outside'}, inplace=True)
dataset = dataset.query("~(Q <= 0)")
dataset = dataset.fillna(method='ffill')
dataset = dataset.drop(['ЧН', 'Код ошибки'], axis=1)
dataset.head()


# In[4]:


dataset.corr()[['Q']].sort_values('Q').T


# In[5]:


# X = dataset[[col for col in dataset.columns if col != 'Q']]
X = dataset[['T_home','T_Outside','M1', 'M2', 'P2', 'T1']]

y = dataset['Q']


# In[6]:


# разделить данные на обучающий набор и временный набор с помощью sklearn.model_selection.traing_test_split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)


# In[7]:


# беру оставшиеся 20% данных в X_tmp, y_tmp и разделю их поровну
X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)

X_train.shape, X_test.shape, X_val.shape
print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))
print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))
print("Testing instances    {}, Testing features    {}".format(X_test.shape[0], X_test.shape[1]))


# In[8]:


feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]


# In[9]:


def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=400):
    return tf.estimator.inputs.pandas_input_fn(x=X,
                                               y=y,
                                               num_epochs=num_epochs,
                                               shuffle=shuffle,
                                               batch_size=batch_size)


# In[29]:


regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[50, 50])
evaluations = []
STEPS = 400
for i in range(300):
    regressor.train(input_fn=wx_input_fn(X_train, y=y_train), steps=STEPS)
    evaluations.append(regressor.evaluate(input_fn=wx_input_fn(X_val,
                                                               y_val,
                                                               num_epochs=1,
                                                               shuffle=False)))


# In[11]:


from sklearn import metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


# вручную установливаю параметры фигуры и соответствующий размер
plt.rcParams['figure.figsize'] = [14, 10]

loss_values = [ev['loss'] for ev in evaluations]
training_steps = [ev['global_step'] for ev in evaluations]

plt.scatter(x=training_steps, y=loss_values)
plt.xlabel('Training steps (Epochs = steps / 2)')
plt.ylabel('Loss (SSE)')
plt.show()


# In[31]:


pred = regressor.predict(input_fn=wx_input_fn(X_test,
                                              num_epochs=1,
                                              shuffle=False))
predictions = np.array([p['predictions'][0] for p in pred])

print("The Explained Variance: %.2f" % explained_variance_score(
                                            y_test, predictions))  
print("The Mean Absolute Error: %.4f гКалл" % mean_absolute_error(
                                            y_test, predictions))  
print("The Median Absolute Error: %.4f гКалл" % median_absolute_error(
                                            y_test, predictions))


# In[32]:


y1 = predictions[0:24]
y2 = y_test[0:24]

x = list(range(0,24))


fig, ax = plt.subplots()


ax.plot(x, y1, label = 'Q predict')
ax.plot(x, y2, label = 'Q test')

ax.legend()
ax.grid()



ax.set_xlabel('Время')
ax.set_ylabel('Q, ГКал')
ax.set_title('Изменение теплопотребления в течение суток')

fig.set_figwidth(16)
fig.set_figheight(8)

plt.show()


# In[33]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
df


# In[34]:


##Проверка на реальных данных
dataset1 = pd.read_excel('David28(11.04-18.04).xls')

dataset1.drop([0,1], inplace=True)
dataset1 = dataset1.drop(['ЧН', 'Код ошибки'], axis=1)
dataset1.rename(columns={'Время': 'Time', 'Tвн': 'T_home', 'Tнв':'T_Outside'}, inplace=True)
dataset1 = dataset1.query("~(Q <= 0)")
dataset1 = dataset1.fillna(method='ffill')
dataset1.head()

dataset1.describe().T


# In[36]:


X_dtest = dataset1[['T_home','T_Outside','M1', 'M2', 'P2', 'T1']]
y_dtest = dataset1['Q']
Y_vnesh = dataset1[['T_Outside']]


# In[37]:


## загрузка обученной модели 
pred1 = regressor.predict(input_fn=wx_input_fn(X_dtest,
                                              num_epochs=1,
                                              shuffle=False))


predictions1 = np.array([p['predictions'][0] for p in pred1])


# In[38]:


print ("R Squared using built-in function: ", r2_score( y_dtest, predictions1))


# In[39]:


print("The Explained Variance: %.2f" % explained_variance_score(
                                            y_dtest, predictions1))  
print("The Mean Absolute Error: %.4f degrees Celcius" % mean_absolute_error(
                                            y_dtest, predictions1))  
print("The Median Absolute Error: %.4f degrees Celcius" % median_absolute_error(
                                            y_dtest, predictions1))

print('Mean Absolute Error: %.6f' % metrics.mean_absolute_error(
                                            y_dtest, predictions1))
print('Mean Squared Error: %.6f' % metrics.mean_squared_error(
                                            y_dtest, predictions1))
print('Root Mean Squared Error: %.6f' % np.sqrt(metrics.mean_squared_error(
                                            y_dtest, predictions1)))


# In[40]:


y1 = predictions1[0:24]
x = list(range(0,24))
y2 = y_dtest[0:24]

fig, ax = plt.subplots()


ax.plot(x, y1, label = 'Виртуальное регулирование графика подачи тепла', color ='g')
ax.plot(x, y2, label = 'Регестрируемое регулирование графика подачи тепла', color ='r')

ax.legend(loc = 'upper right', fontsize = 15)
ax.grid()

ax.set_xlabel('Время', fontsize =15)
ax.set_ylabel('Q, ГКал', fontsize =15)
ax.set_title('Изменение теплопотребления в течение суток', fontsize =15)

fig.set_figwidth(16)
fig.set_figheight(8)

plt.show()


# In[28]:


df = pd.DataFrame({'Qрег': y_dtest, 'Qм': predictions1})
df1 = df.head(25)
df1


# In[ ]:


##Гистограмма
df1.plot(kind = 'bar', figsize=(10,8))
plt.grid(which='major', linestyle ='-', linewidth='0.5', color = 'black')
plt.grid(which='minor', linestyle =':', linewidth='0.5', color = 'r')
plt.show()

