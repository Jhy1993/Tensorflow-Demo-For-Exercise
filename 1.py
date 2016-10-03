a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.mul(a, b)
with tf.Session() as sess:
    print "add is %d" % sess.run(add, feed_dict={a: 2, b: 3})


mat1 = tf.constant([[3, 3]])
mat2 = tf.constant([[2], [2]])
product = tf.matmul(mat1, mat2)
with tf.Session() as sess:
    re = sess.run(product)
    print re


