
import numpy as np 
import tensorflow as tf 


print(tf.constant([[1., 2., 3.], [4., 5., 6.]]))
print(tf.constant(42))

t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(t @ tf.transpose(t))

a = np.array([2., 4., 5.])
print(tf.constant(a))
print(t.numpy())
print(tf.square(a))
print(np.square(t))

v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
v.assign(2 * v)
v[:, 2].assign([0., 1.])
v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100., 200.])