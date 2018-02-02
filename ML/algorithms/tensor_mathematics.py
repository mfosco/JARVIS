import tensorflow as tf
import numpy as np
import math


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

# addition and subtraction
print(sess.run(A+B))
print(sess.run(B-B))
# multiplication
print(sess.run(tf.matmul(B, identity_matrix))) #can specify whether or not to transpose arguments before multiplication in matmul()
# transpose
print(sess.run(tf.transpose(C))) # re initializing gives us different values from before
# determinant:
print(sess.run(tf.matrix_determinant(D)))
# Inverse
print(sess.run(tf.matrix_inverse(D)))

# Cholesky
print(sess.run(tf.cholesky(identity_matrix)))
# eigen values and eigenvectors, returns things as eigenvalues, matrix of eigenvectors
print(sess.run(tf.self_adjoint_eig(D)))

# note, div() returns the same type as the inputs (so it returns the floor of the division if inputs are ints)
# truediv() will return ints turned into floats
print(sess.run(tf.div(3,4))) # ret 0
print(sess.run(tf.truediv(3,4))) # ret .75

# if have floats and want integer division, use "floordiv()"
print(sess.run(tf.floordiv(3.0, 4.0)))

# don't forget mod:
print(sess.run(tf.mod(22.0, 5.0)))

# cross product only defived for 2 three-dimensional vectors
print(sess.run(tf.cross([1., 0., 0.], [0., 1., 0.])))

'''
list of math functions in tf:
abs()
ceil()
cos()
exp()
floor()
inv()   # multiplicative inverse
log()
maximum()   # element-wise max of 2 tensors
minimum()   # element-wise min of 2 tensors
neg()       # negative of one input tensor
pow()       # first tensor raided to the second tensor element-wise
round()     # round 1 input tensor
rsqrt()     # 1 over the square root of one tensor
sign()      # returns -1, 0 or 1 depending on the sign of the tensor
sin() 
sqrt()      # squre root of one input tensor
square()    # square of one input tensor
'''

'''
Special ML functions:
digamma()   # Psi function, the derivative of the lgamma() function
erf()       # Gaussian error function, element-wise of one tensor
erfc()      # complimentary error function of one tensor
igamma()    # lower regularized incomplete gamma function
lbeta()     # natural log of the absolute value of the beta function
lgamma()    # natural log of the absolute value of the gamma function
squared_differece()     # computes the square of the differences between 2 tensors
'''

# making a composite function
# tan(pi/4) = 1
print(sess.run(tf.div(tf.sin(math.pi/4.), tf.cos(math.pi/4.))))

# 3x^2 - x + 10
def custom_polynomial(value):
    return(tf.subtract(3 * tf.square(value), value) + 10)

print(sess.run(custom_polynomial(11)))

#########################################
# Activation functions
# ReLU
print(sess.run(tf.nn.relu([-3., 3., 10., -4])))

# capping relu function at some value, 6 in this case
print(sess.run(tf.nn.relu6([-3., 3., 10.])))

# that vanishing sigmoid
print(sess.run(tf.nn.sigmoid([-1., 0., 1.])))

# tanh ie ((exp(x)-exp(-x))/(exp(x)+exp(-x))
print(sess.run(tf.nn.tanh([-1., 0., 1.])))

#softsign function ie x/(abs(x) + 1, so basically a continuous version of sin
print(sess.run(tf.nn.softsign([-1., 0., -1.])))

# softplus, ie smooth version of ReLU, form: log(exp(x) + 1)
print(sess.run(tf.nn.softplus([-1., 0., -1.])))

# Exponential Linear Unit (ELU)
print(sess.run(tf.nn.elu([-1., 0., 1.])))
