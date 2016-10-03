
import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hots=True)
x_data = tf.placeholder('float', [None, 784])
y_data = tf.placeholder('float', [None, 10])
W = tf.Variable(tf.zeros([10, 784]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(W, x_data) + b)

cross_entropy = -tf.reduce_sum(y_data * tf.log(y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(cross_entropy)

init = tf.initialize_all_variables

sess = tf.Session()
sees.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(tranin_step, 
        feed_dict={x: batch_xs, y_data: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sees.run(accuracy, 
    feed_dict={x: mnist.test.images, y: mnist.test.labels})



