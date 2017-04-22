---
layout: 		post
title: 		"Intro to TrnsorFlow"
subtitle :	"Udacity"
date: 			2016-12-5 17:31:59
author: 		"Mr. freedom"
header-img: 	"/img/post-bg-2015.jpg"
tags:
	 - python
	 - Tensorflow
---

## Hello, Tensor World!

```python
import tensorflow as tf

# Create TensorFlow object called hello_constant
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
```

## Tensor

In TensorFlow, data isn’t stored as integers, floats, or strings. These values are encapsulated in an object called a tensor. In the case of ```hello_constant = tf.constant('Hello World!')```, hello_constant is a 0-dimensional string tensor, but tensors come in a variety of sizes as shown below:

```python
# A is a 0-dimensional int32 tensor
A = tf.constant(1234) 
# B is a 1-dimensional int32 tensor
B = tf.constant([123,456,789]) 
 # C is a 2-dimensional int32 tensor
C = tf.constant([ [123,456,789], [222,333,444] ])
```

## Session

TensorFlow’s api is built around the idea of a computational graph, a way of visualizing a mathematical process which you learned about in the MiniFlow lesson. Let’s take the TensorFlow code you ran and turn that into a graph:![That's it!](https://s3.cn-north-1.amazonaws.com.cn/u-img/33f8ba4e-26f9-4f69-8fd6-7e0500fe4117)
A "TensorFlow Session", as shown above, is an environment for running a graph. The session is in charge of allocating the operations to GPU(s) and/or CPU(s), including remote machines. Let’s see how you use it.

```python
with tf.Session() as sess:
    output = sess.run(hello_constant)
```

## Tensorflow Input
### tf.placeholder()

Sadly you can’t just set ```x``` to your dataset and put it in TensorFlow, because over time you'll want your TensorFlow model to take in different datasets with different parameters. You need ```tf.placeholder()```!

```tf.placeholder()``` returns a tensor that gets its value from data passed to the ```tf.session.run()``` function, allowing you to set the input right before the session runs.

### session's feed_dict()

```python
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
```

Use the feed_dict parameter in tf.session.run() to set the placeholder tensor. The above example shows the tensor x being set to the string "Hello, world". It's also possible to set more than one tensor using feed_dict as shown below.

```python
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})

```

**Note:**if the data passed to the ```feed_dict``` doesn’t match the tensor type and can’t be cast into the tensor type, you’ll get the error ```“ValueError: invalid literal for...”```.

## TensorFlow Math
### Addition

```python
x = th.add(5, 2)  # 7
``` 

### Subtraction and Multiplication and Division

```python
x = tf.subtract(10, 4) # 6
y = tf.multiply(2, 5)  # 10
z = tf.divide(4, 2)  # 2
```

### converting types

```python
tf.subtract(tf.constant(2.0),tf.constant(1))  # Fails with ValueError: Tensor conversion requested dtype float32 
```

That's because the constant ```1``` is an integer but the constant ```2.0``` is a floating point value and ```subtract``` expects them to match.

```python
tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   # 1
```

## TensorFlow Linear Function
### tf.Variable()

> This leaves out ```tf.placeholder()``` and ```tf.constant()```, since those Tensors can't be modified. This is where tf.Variable class comes in.

```python
x = tf.Variable()
```

The ```tf.Variable``` class creates a tensor with an initial value that can be modified, much like a normal Python variable. This tensor stores its state in the session, so you must initialize the state of the tensor manually. You'll use the ```tf.global_variables_initializer()``` function to initialize the state of all the Variable tensors.

#### Initialization

```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

Similarly, choosing weights from a normal distribution prevents any one weight from overwhelming other weights. You'll use the ```tf.truncated_normal()``` function to generate random numbers from a normal distribution.

### tf.truncated_normal()

```python
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
```

The ```tf.truncated_normal()``` function returns a tensor with random values from a normal distribution whose magnitude is no more than 2 standard deviations from the mean.

Since the weights are already helping prevent the model from getting stuck, you don't need to randomize the bias. Let's use the simplest solution, setting the bias to 0.

### tf.zeros()

```python
n_labels = 5
bias = tf.Variable(tf.zeros(n_labels))

```

The ```tf.zeros()``` function returns a tensor with all zeros.

## TensorFlow Softmax
![Woo~](https://s3.cn-north-1.amazonaws.com.cn/u-img/e249ce82-8329-45d3-a91c-7b85f18149ed)
### TensorFlow Softmax

We're using TensorFlow to build neural networks and, appropriately, there's a function for calculating softmax.

```python
x = tf.nn.softmax([2.0, 1.0, 0.2])
```

## One-Hot Encoding
we set only one probality close to 1 as 1,others as 0.

## TensorFlow Cross Entropy
![hh](https://s3.cn-north-1.amazonaws.com.cn/u-img/8e5b99b1-48df-4d80-b394-9ff15c6eba89)

To create a cross entropy function in TensorFlow, you'll need to use two new functions:

*  ```tf.reduce_sum()```
*  ```tf.log()```

### Reduce sum

```python
x = tf.reduce_sum([1, 2, 3, 4, 5])  # 15
```

The ```tf.reduce_sum()``` function takes an array of numbers and sums them together.

### Natural Log

```python
x = tf.log(100)  # 4.60517
```

This function does exactly what you would expect it to do. tf.log() takes the natural log of a number.The natural number is ```e```.

## Mini-batch

Mini-batching is a technique for training on subsets of the dataset instead of all the data at one time. 

### batches()

```python
# 4 Samples of features
example_features = [
    ['F11','F12','F13','F14'],
    ['F21','F22','F23','F24'],
    ['F31','F32','F33','F34'],
    ['F41','F42','F43','F44']]
# 4 Samples of labels
example_labels = [
    ['L11','L12'],
    ['L21','L22'],
    ['L31','L32'],
    ['L41','L42']]
    
example_batches = batches(3, example_features, example_labels)
```

The ```example_batches``` variable would be the following:

```python
[
    # 2 batches:
    #   First is a batch of size 3.
    #   Second is a batch of size 1
    [
        # First Batch is size 3
        [
            # 3 samples of features.
            # There are 4 features per sample.
            ['F11', 'F12', 'F13', 'F14'],
            ['F21', 'F22', 'F23', 'F24'],
            ['F31', 'F32', 'F33', 'F34']
        ], [
            # 3 samples of labels.
            # There are 2 labels per sample.
            ['L11', 'L12'],
            ['L21', 'L22'],
            ['L31', 'L32']
        ]
    ], [
        # Second Batch is size 1.
        # Since batch size is 3, there is only one sample left from the 4 samples.
        [
            # 1 sample of features.
            ['F41', 'F42', 'F43', 'F44']
        ], [
            # 1 sample of labels.
            ['L41', 'L42']
        ]
    ]
]
```

## Epochs

An epoch is a single forward and backward pass of the whole dataset. This is used to increase the accuracy of the model without requiring more data. This section will cover epochs in TensorFlow and how to choose the right number of epochs.

The following TensorFlow code trains a model using 10 epochs.

```python
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from helper import batches  # Helper function created in Mini-batching section


def print_epoch_stats(epoch_i, sess, last_features, last_labels):
    """
    Print cost and validation accuracy of an epoch
    """
    current_cost = sess.run(
        cost,
        feed_dict={features: last_features, labels: last_labels})
    valid_accuracy = sess.run(
        accuracy,
        feed_dict={features: valid_features, labels: valid_labels})
    print('Epoch: {:<4} - Cost: {:<8.3} Valid Accuracy: {:<5.3}'.format(
        epoch_i,
        current_cost,
        valid_accuracy))

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
valid_features = mnist.validation.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
valid_labels = mnist.validation.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Features and Labels
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

# Logits - xW + b
logits = tf.add(tf.matmul(features, weights), bias)

# Define loss and optimizer
learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Calculate accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

batch_size = 128
epochs = 10
learn_rate = 0.001

train_batches = batches(batch_size, train_features, train_labels)

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch_i in range(epochs):

        # Loop over all batches
        for batch_features, batch_labels in train_batches:
            train_feed_dict = {
                features: batch_features,
                labels: batch_labels,
                learning_rate: learn_rate}
            sess.run(optimizer, feed_dict=train_feed_dict)

        # Print cost and validation accuracy of an epoch
        print_epoch_stats(epoch_i, sess, batch_features, batch_labels)

    # Calculate accuracy for test dataset
    test_accuracy = sess.run(
        accuracy,
        feed_dict={features: test_features, labels: test_labels})

print('Test Accuracy: {}'.format(test_accuracy))
```









