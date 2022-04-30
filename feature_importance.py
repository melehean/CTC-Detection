import matplotlib.pyplot as plt

from data import Data
import tensorflow as tf
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.preprocessing import OneHotEncoder
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data_filepath = "data/SC_integration/counts_ctc_simulated_123_5k.tsv"
true_results_filepath = "data/SC_integration/ids_ctc_simulated_123_5k.tsv"
train_indices_filepath = "data/SC_integration/train_indices.npy"
test_indices_filepath = "data/SC_integration/test_indices.npy"

data_object = Data(data_filepath, true_results_filepath)

train_data, test_data, train_true_results, test_true_results =  data_object.load_train_test_split(train_indices_filepath, test_indices_filepath)
#print(type(train_data))
#print(train_data.columns.values.tolist())
#print(type(train_data[0][0]))
#print(np.float32)
#print(np.array(train_data,np.float32))
data=train_data
train_data, test_data, train_true_results, test_true_results=np.array(train_data,np.float32),np.array(test_data,np.float32),np.array(train_true_results,np.int32),np.array(test_true_results,np.int32)


# num_classes = 2
# num_features = 2000
#
# learning_rate = 0.0001
# training_steps = 10000
# batch_size = train_data.shape[0]
# display_step = training_steps/100
#
# train_data=tf.data.Dataset.from_tensor_slices((train_data,train_true_results))
# train_data=train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
#
# W = tf.Variable(tf.ones([num_features, num_classes]),tf.float32, name="weight")
# b = tf.Variable(tf.zeros([num_classes]),tf.float32, name="bias")
#
# # Logistic regression (Wx + b).
#
# def logistic_regression(x):
#
#     # Apply softmax to normalize the logits to a probability distribution.
#
#     return tf.nn.softmax(tf.matmul(x, W) + b)
#
# # Cross-Entropy loss function.
#
# def cross_entropy(y_pred, y_true):
#
#     # Encode label to a one hot vector.
#
#     y_true = tf.one_hot(y_true, depth=num_classes)
#
#     # Clip prediction values to avoid log(0) error.
#
#     y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
#
#     # Compute cross-entropy.
#
#     return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))
#
# # Accuracy metric.
#
# def accuracy(y_pred, y_true):
#     # Predicted class is the index of the highest score in prediction vector (i.e. argmax).
#     correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
#     return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# # Stochastic gradient descent optimizer.
#
# optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
#
#
# # Optimization process.
#
# def run_optimization(x, y):
#     # Wrap computation inside a GradientTape for automatic differentiation.
#
#     with tf.GradientTape() as g:
#         pred = logistic_regression(x)
#
#         loss = cross_entropy(pred, y)
#
#     # Compute gradients.
#
#     gradients = g.gradient(loss, [W, b])
#
#     # Update W and b following gradients.
#
#     optimizer.apply_gradients(zip(gradients, [W, b]))
#
#
# for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
#     # Run the optimization to update W and b values.
#
#     run_optimization(batch_x, batch_y)
#
#     if step % display_step == 0:
#         pred = logistic_regression(batch_x)
#
#         loss = cross_entropy(pred, batch_y)
#
#         acc = accuracy(pred, batch_y)
#
#         print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
#
#
# # Test model on validation set.
# pred = logistic_regression(test_data.astype('float32'))
# print("Test Accuracy: %f" % accuracy(pred, test_true_results))

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
#train_data=np.hstack((train_data,np.ones((train_data.shape[0],1),np.float32)))
#test_data=np.hstack((test_data,np.ones((test_data.shape[0],1),np.float32)))
#print(train_data)
regularization=0.02
clf = LogisticRegression(random_state=0,class_weight='balanced',penalty='l1',C=regularization,solver='liblinear')
clf.fit(train_data, train_true_results)
print("Train accuracy: "+str(clf.score(train_data, train_true_results)))
print("Test accuracy: "+str(clf.score(test_data, test_true_results)))

#print(clf.get_params())
#print(clf.coef_)
#print(clf.coef_[:-1])
#print(clf.feature_names_in_)

zipped=sorted(zip(clf.coef_.tolist()[0],data.columns.values.tolist()))

importance_threshold=0.02
genes_that_matter=list(filter(lambda x: abs(x[0])>importance_threshold,zipped))
print("Number of genes that matter: "+str(len(genes_that_matter))+" out of "+str(len(zipped))+"  ("+str(100*len(genes_that_matter)/len(zipped))+"%)")

#plt.figure(1)
genes_importances=list(zip(*genes_that_matter))#list(zip(*zipped))
plt.plot(genes_importances[1],genes_importances[0])
plt.title("Importances of genes that matter in prediction ("+str(100*len(genes_that_matter)/len(zipped))+"% of all genes)"+"\nPositive values mean genes, which predict cancer,\nnegative values are genes, which predict absence of cancer")
#plt.legend("Gene influence on prediction")
plt.xticks(rotation=90)
#plt.tick_params(axis='both', which='minor', labelsize=5)
plt.tick_params(axis='x', which='major', labelsize=6)
plt.show()

