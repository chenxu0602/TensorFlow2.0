import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

SIZE=1000
a, b1, b2 = 2.0, 1.0, 3.0
x = np.random.randn(SIZE, 1)
y1 = a * x + b1
y2 = a * x + b2

theta = np.random.randn(2, 1)
x_b = np.c_[np.ones((SIZE, 1)), x]
lr = 0.01

for epoch in range(200):
    # gradients = 2 / SIZE * x_b.T.dot(x_b.dot(theta) - y2)
    grad1 = 2 / SIZE * x_b.T.dot(np.maximum(x_b.dot(theta) - y2, 0))
    grad2 = 2 / SIZE * x_b.T.dot(np.minimum(x_b.dot(theta) - y1, 0))
    gradients = grad1 + grad2
    theta -= lr * gradients