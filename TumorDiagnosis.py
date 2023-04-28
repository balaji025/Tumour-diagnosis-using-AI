#importing pandas to read the csv file
import pandas as pd
import numpy as np
dataset = pd.read_csv('cancer.csv')

x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])     #x as every column except diagnosis column
y=dataset["diagnosis(1=m, 0=b)"]  #y as the diagnosis column

#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

#building a neural network for the ai using tensorflow's keras
import tensorflow as tf

#using a sequential model
model = tf.keras.models.Sequential()

# The core idea of Sequential API is simply arranging the Keras layers in a sequential order and so, it is called Sequential API. 
# Most of the ANN also has layers in sequential order and the data flows from one layer
# to another layer in the given order until the data finally reaches the output layer.

model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid'))
#model.add(tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#model.add(tf.keras.Input(shape=(x_train.shape[1],))) 
#model.add(tf.keras.layers.Dense(128, activation='sigmoid')) 
#model.add(tf.keras.layers.Dense(256, activation='sigmoid')) 
#model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
# Computes the cross-entropy loss between true labels and predicted labels.
# Accuracy metric: Calculates how often predictions equal labels.
# labels = training set
# pred = predictions made

model.fit(x_train, y_train, epochs=1000)

model.evaluate(x_test, y_test)

# Evaluation is a process during development of the model to check whether the model is best fit for 
# the given problem and corresponding data. Keras model provides a function, evaluate which does the evaluation of the model.

# to make prediction for a given input
features = np.array([[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,
                      0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]])
j=model.predict(features)
print(j)
