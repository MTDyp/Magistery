#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import explained_variance_score,     mean_absolute_error,     median_absolute_error
from sklearn.model_selection import train_test_split


# In[57]:


import os

cwd = os.getcwd()
os.chdir("C:\\Users\\MPC\\Downloads")
cwd
dataset = pd.read_excel('Davidova28nT(12.19-05.20).xls')

# execute the describe() function and transpose the output so that it doesn't overflow the width of the screen
dataset.describe().T


# In[58]:


dataset.drop([0,1], inplace=True)
dataset.rename(columns={'Время': 'Time', 'Tвн': 'T_home, C', 'Tнв':'T_Inside'}, inplace=True)
dataset = dataset.query("~(Q <= 0)")
dataset = dataset.fillna(method='ffill')
dataset.head()


# In[59]:


dataset.rename(columns={'T_home, C': 'T_home'}, inplace=True)


# In[45]:


dataset.describe().T


# In[60]:


dataset.corr()[['Q']].sort_values('Q')


# In[34]:


dataset.info()


# In[61]:


David = dataset.copy()


# In[62]:


# First drop the maxtempm and mintempm from the dataframe
dataset = dataset.drop(['ЧН', 'Код ошибки'], axis=1)

# X will be a pandas dataframe of all columns except meantempm
X = dataset[[col for col in dataset.columns if col != 'Q']]

# y will be a pandas series of the meantempm
y = dataset['Q']


# In[63]:


# split data into training set and a temporary set using sklearn.model_selection.traing_test_split
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=23)


# In[64]:


# take the remaining 20% of data in X_tmp, y_tmp and split them evenly
X_test, X_val, y_test, y_val = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=23)

X_train.shape, X_test.shape, X_val.shape
print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))
print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))
print("Testing instances    {}, Testing features    {}".format(X_test.shape[0], X_test.shape[1]))


# In[65]:


feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]


# In[66]:


regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[50, 50],
                                      model_dir='tf_wx_model')


# In[67]:


def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=400):
    return tf.estimator.inputs.pandas_input_fn(x=X,
                                               y=y,
                                               num_epochs=num_epochs,
                                               shuffle=shuffle,
                                               batch_size=batch_size)


# In[68]:


evaluations = []
STEPS = 400
for i in range(100):
    regressor.train(input_fn=wx_input_fn(X_train, y=y_train), steps=STEPS)
    evaluations.append(regressor.evaluate(input_fn=wx_input_fn(X_val,
                                                               y_val,
                                                               num_epochs=1,
                                                               shuffle=False)))


# In[69]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# manually set the parameters of the figure to and appropriate size
plt.rcParams['figure.figsize'] = [14, 10]

loss_values = [ev['loss'] for ev in evaluations]
training_steps = [ev['global_step'] for ev in evaluations]

plt.scatter(x=training_steps, y=loss_values)
plt.xlabel('Training steps (Epochs = steps / 2)')
plt.ylabel('Loss (SSE)')
plt.show()


# In[71]:


pred = regressor.predict(input_fn=wx_input_fn(X_test,
                                              num_epochs=1,
                                              shuffle=False))
predictions = np.array([p['predictions'][0] for p in pred])

print("The Explained Variance: %.2f" % explained_variance_score(
                                            y_test, predictions))  
print("The Mean Absolute Error: %.4f degrees Celcius" % mean_absolute_error(
                                            y_test, predictions))  
print("The Median Absolute Error: %.4f degrees Celcius" % median_absolute_error(
                                            y_test, predictions))


# In[105]:


# Qnorm = dataset['Q']*((18-dataset['T_Inside'])/(18+24))
# Qnorm1 = Qnorm.head(24)
y1 = predictions[0:24]
x = list(range(0,24))
# Q1 =dataset['Q'].head(24)
# y1 = Q1.tolist()

# Q2 = y.tolist()
y2 = y_test[0:24]

# Q3 = y_test.tolist()
# y3 = Q3[0:24]
fig, ax = plt.subplots()

ax.plot(x, y1, x, y2)
ax.grid()

# f_nom = 22.5 # Номинальная температура, которую следует поддерживать
# xmin = table.values [0,0] # Начало оси x
# xmax = '23:59:59' # Конец оси x
# ax.hlines(f_nom, xmin, xmax, colors = 'r')


# ymin = table.values [0,1] # Начало оси y
# ymax = 28 # Конец оси x


ax.set_xlabel('Время')
ax.set_ylabel('Q, ГКал')
ax.set_title('Изменение теплопотребления в течение суток')

fig.set_figwidth(16)
fig.set_figheight(8)

plt.show()


# In[103]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
df

