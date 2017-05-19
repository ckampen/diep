import tensorflow as tf

Y_ = tf.placeholder(tf.float32, [None, 10])
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
L1D = 6  # first convolutional layer output depth
L2D = 12  # second convolutional layer output depth
L3D = 24  # third convolutional layer
L4D = 200  # fully connected layer

def biasVar(output_depth):
    return tf.Variable(tf.ones([output_depth]) / 10)

def weightVar(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def conv2d(input_layer, w, b, stride=1):
    print("using stride: %d" % stride)
    strides = [1, stride, stride, 1]
    cnv = tf.nn.conv2d(input_layer, w, strides=strides, padding='SAME')
    return tf.nn.relu(cnv + b)

def matmul(input_layer, w, b):
    dotpr = tf.matmul(input_layer, w)
    return tf.nn.relu(dotpr + b)

#Network model

#Convolutional layers

#Layer 1: convolution 28x28 -> 28x28
W1 = weightVar([6,6,1,L1D])
B1 = biasVar(L1D)
OutputLayer1 = conv2d(X, W1, B1)
stride = 1
#Layer 2 : 28x28 -> 14x14 (stride = 2: (28/2)x(28/2) )
stride = 2
W2 = weightVar([5,5,L1D,L2D])
B2 = biasVar(L2D)
OutputLayer2 = conv2d(OutputLayer1, W2, B2, stride=2)
#layer 3 : 14x14 -> 7x7
W3 = weightVar([4,4,L2D,L3D])
B3 = biasVar(L3D)
OutputLayer3 = conv2d(OutputLayer2, W3, B3, stride=2)
#Layer 4 : fully connected (we need to flatten the output of Layer 3 first)
flatShape = 7 * 7 * L3D # size of the output of layer 3 (width * hight * depth)
#flatten the output of Layer 3
YY = tf.reshape(OutputLayer3, shape=[-1, flatShape])
#Layer 4
W4 = weightVar([flatShape, L4D])
B4 = biasVar(L4D)
OutputLayer4 = matmul(YY, W4, B4)
# OutputLayer4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
#Dropout
keep_prob = tf.placeholder(tf.float32)
L4Drop = tf.nn.dropout(OutputLayer4, keep_prob)
#layer 5 : output layer
outputNeuronCount = 10
W5 = weightVar([L4D, outputNeuronCount])
B5 = biasVar(outputNeuronCount)
# logits: the inverse of the logistic function
Ylogits = tf.matmul(L4Drop, W5) + B5

#softmax for classification (probability score)
Y = tf.nn.softmax(Ylogits)
