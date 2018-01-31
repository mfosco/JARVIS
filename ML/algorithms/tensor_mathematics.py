import tensorflow as tf
import numpy as np

# printing a regular tensor, oh boy

sess    = tf.Session()
tens1   = tf.constant([[[1,2], [2,3]], [[3,4], [5,6]]])

row_dim = 4
col_dim = 5
# zero filled vector
zero_tsr    = tf.zeros([row_dim, col_dim])

ones_tsr    = tf.ones([row_dim, col_dim])

# constant filled tensor
filled_tsr  = tf.fill([row_dim, col_dim], 42) # the meaning of life

# tensor out of existing constant:
constant_tsr    = tf.constant([1,2,3])

# tf.constant can broadcast a value into an array, mimicking the behavior
# of tf.fill() by writing tf.constant(42, [row_dim, col_dim])

zeros_similar   = tf.zeros_like(constant_tsr)
ones_similar    = tf.ones_like(constant_tsr)
linear_tsr  = tf.linspace(start=0.0, stop=1.0, num=3)
integer_seq_tsr     = tf.range(start=6, limit=15, delta=3)
#print(sess.run(tens1)[1,1,0])

# x in [0, 1)
randunif_tsr    = tf.random_uniform([row_dim, col_dim], minval=0, maxval=1)
randnorm_tsr    = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)

print(linear_tsr)
print(integer_seq_tsr)
sess.run(linear_tsr)
sess.run(integer_seq_tsr)

# truncated normal below always returns values within 2 standard deviations of the mean
runcnorm_tsr    = tf.truncated_normal([row_dim, col_dim], mean=0.0, stddev=1.0)
my_var  = tf.Variable(tf.zeros([2,3]))
initialize_op   = tf.global_variables_initializer()
sess.run(initialize_op)

x   = tf.placeholder(tf.float32, shape=[2,2])
y   = tf.identity(x)
x_vals  = np.random.rand(2,2)
sess.run(y, feed_dict={x: x_vals})

# initializing variables in the order that we want
first_var   = tf.Variable(tf.zeros([2,3]))
sess.run(first_var.initializer)
second_var  = tf.Variable(tf.zeros_like(first_var))
# depends on first_var
sess.run(second_var.initializer)

# matrices now yo
identity_matrix     = tf.diag([1.0, 1.0, 1.0])
A   = tf.truncated_normal([2,3])
B   = tf.fill([2,3], 5.0)
C   = tf.random_uniform([3,2])
D   = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print(sess.run(identity_matrix))
print(sess.run(A))
print(sess.run(B))
print(sess.run(C))
print(sess.run(D))
