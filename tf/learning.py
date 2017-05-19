
import tensorflow as tf
import math

from neural_net import X, Ylogits, Y, Y_, keep_prob
from data import MnistData
tf.set_random_seed(0)

lr = tf.placeholder(tf.float32)

dataprovider = MnistData()
print(dataprovider.size())
#Network Training functions

#cross entropy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data, dataprovider):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = dataprovider.next_batch(100)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    decay = (max_learning_rate - min_learning_rate) * math.exp(-1/decay_speed)
    learning_rate = min_learning_rate + decay

    if update_train_data:
        input_dict = {X: batch_X, Y_: batch_Y, keep_prob: 1.0}
        a, c = sess.run([accuracy, cross_entropy], input_dict)
        print("%d: accuracy: %f loss: %f (lr:%f)" % (i,a,c,learning_rate))

    # the backpropagation training step
    input_dict = {X: batch_X, Y_: batch_Y, lr: learning_rate, keep_prob: 0.75}
    sess.run(train_step, input_dict)

for i in range(10000+1):
    training_step(i, i % 100 == 0, i % 20 == 0, dataprovider)

input_dict = {X: dataprovider.test_X, Y_: dataprovider.test_Y, keep_prob: 1.0}
a, c = sess.run([accuracy, cross_entropy], input_dict)

epoch = i * 100 // dataprovider.size()
print("%d: epoch %d test accuracy: %f test loss %f" % (i,epoch,a,c))

