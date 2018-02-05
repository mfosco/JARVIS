import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sess    = tf.Session()

x_vals  = np.array([1., 3., 5., 7., 9.])
x_data  = tf.placeholder(tf.float32)
m_const     = tf.constant(3.)
my_product  = tf.multiply(x_data, m_const)
for x_val in x_vals:
    print(sess.run(my_product, feed_dict={x_data: x_val}))

#############################
# multiplying matrices
# inputs
my_array    = np.array([[1., 3., 5., 7., 9.],
                        [-2., 0., 2., 4., 6.],
                        [-6., -3., 0., 3., 6.]])
x_vals  = np.array([my_array, my_array + 1])
x_data  = tf.placeholder(tf.float32, shape=(3,5))

# constants
m1  = tf.constant([[1.], [0.], [-1], [2.], [4.]])
m2  = tf.constant([[2.]])
a1  = tf.constant([[10.]])

prod1   = tf.matmul(x_data, m1)
prod2   = tf.matmul(prod1, m2)
add1    = tf.add(prod2,a1)

for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data:x_val}))

# to have data placeholder have an unknown # of columns, do the following:
# x_data    = tf.placeholder(tf.float32, shape=(3,None))

# connecting layers:
# sample 4 x 4 image
x_shape     = [1,4,4,1]
x_val   = np.random.uniform(size=x_shape)

# need a placeholder
x_data  = tf.placeholder(tf.float32, shape=x_shape)
my_filter   = tf.constant(.25, shape=[2,2,1,1])
my_strides  = [1, 2, 2, 1]
mov_avg_layer   = tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_Avg_Window')

def custom_layer(input_matrix):
    input_matrix_squeezed   = tf.squeeze(input_matrix)
    A   = tf.constant([[1., 2.], [-1., 3.]])
    b   = tf.constant(1., shape=[2,2])
    temp1   = tf.matmul(A, input_matrix_squeezed)
    temp    = tf.add(temp1, b) # ie Ax + b
    return tf.sigmoid(temp)

with tf.name_scope('Custom_Layer') as scope:
    custom_layer1   = custom_layer(mov_avg_layer)

print(sess.run(custom_layer1, feed_dict={x_data: x_val}))

# implementing loss functions
x_vals  = tf.linspace(-1., 1., 500)
target  = tf.constant(0.)

l2_y_vals   = tf.square(target - x_vals)
l2_y_out    = sess.run(l2_y_vals)

l1_y_vals   = tf.abs(target - x_vals)
l1_y_out    = sess.run(l1_y_vals)

delta1  = tf.constant(0.25)
phuber1_y_vals  = tf.multiply(tf.square(delta1), tf.sqrt(1. +
                                                    tf.square((target - x_vals)/delta1)) - 1.)
phuber1_y_out   = sess.run(phuber1_y_vals)
delta2  = tf.constant(5.)
phuber2_y_vals  = tf.multiply(tf.square(delta2), tf.sqrt(1. +
                                                         tf.square((target - x_vals)/delta2)) - 1.)
phuber2_y_out   = sess.run(phuber2_y_vals)

x_vals  = tf.linspace(-3., 5., 500)
target  = tf.constant(1.)
targets     = tf.fill([500,], 1.)

hinge_y_vals    = tf.maximum(0., 1. - tf.multiply(target, x_vals))
hinge_y_out     = sess.run(hinge_y_vals)

