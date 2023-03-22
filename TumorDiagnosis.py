#importing pandas to read the csv file
import pandas as pd
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

#adding the data to the neural network layers
model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#to compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#to train them model
model.fit(x_train, y_train, epochs=1000)

#to test the model
model.evaluate(x_test, y_test)
