# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 22:21:00 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

README:

Reference:
TensorFlow For Machine Intelligence P89
"""
import tensorflow as tf 
g = tf.Graph()

with g.as_default():
    a = tf.mul(2, 3)

g1 = tf.Graph()
with g1.as_default():
    ....
g2 = tf.Graph()
with g2.as_default():
    ...


a = tf.add(2, 5)
b = tf.mul(a, 3)

sess = tf.Session()
re_dict = {a: 11}

sess.run(b, feed_dict=re_dict)

a = tf.constant(5)
sees = tf.Session()
with sess.as_default():
    a.eval()
sess.close()

a = tf.placeholder(tf.int32, shape=[2], name="myinput")
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_C")
d = tf.add(b, c, name="add_D")

input_dict = {a: np.array([5, 3], dtype=np.int32)}
sess.run(d, feed_dict=input_dict)

my_var = tf.Variable(2, name="my_variables")
add = tf.add(5, my_var)

uniform = tf.random_uniform([3, 3, 3], minval=0, maxval=10)
normal = tf.random_uniform([3, 3, 3], mean=0.0, stddev=2.0)

trunc = tf.truncated_normal([2, 2], mean=5.0, stddev=1.0)
random_var = tf.Variable(tf.truncated_normal([2, 2], mean=5.0, stddev=1.0))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#initialize subset of variables
v1 = tf.Variable(0, name="v1")
v2 = tf.Variable(1, name="v2")
init = tf.initialize_all_variables([v1], name="init+v1")
sess = tf.Session()
sess.run(init)
#change value of variable in session
v = tf.Variable(1)
v2 = v.assign(v * 2)# is a operation , not a variable
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
sess.run(v2) # v = 2
sess.run(v2) # v = 4

not_trainable = tf.Variable(0, trainabale=False)
#scope
with tf.name_scope("scope_a"):
    a = tf.add(1, 2, name="a")
    b = tf.mul(a, 2, name="b")
with tf.name_scope("scope_b"):
    c = tf.add(3, 4, name="c")
    d = tf.mul(c, 6, name="D")
e = tf.add(b, d, name="E")
writer = tf.train.SummaryWriter('./name_scope_1', graph=tf.get_default_graph())
writer.close()
tensorboard --logdir='./name_scope_1'


graph = tf.Graph()
with graph.as_default():
    in1 = tf.placeholer(tf.float32, shape=[], name="in1")
    in2 = tf.placeholer(tf.float32, shape=[], name="in2")
    const = tf.constant(3, dtype=tf.float32, name="static_value")
    with tf.name_scope("trans"):
        with tf.name_scope("A"):
            A_mul = tf.mul(in1, const)
            A_out = tf.sub(A_mul, in_1)
        with tf.name_scope("B"):
            B_mul = tf.mul(in2, const)
            B_out = tf.sun(B_mul, in2)




graph = tf.Graph()
with graph.as_default():

    with tf.name_scope("variables"):
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        total_output = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="total_output")

    with tf.name_scope("trans"):

        #input layer
        with tf.name_scope("input"):
            a = tf.placeholer(tf.float32, shape=[None], name="input_a")

        # middle layer
        with tf.name_scope("middle_layer"):
            b = tf.reduce_prod(a, name="prod_b")
            c = tf.reduce_sum(a, name="sum_c")

        #output
        with tf.name_scope("output"):
            output = tf.add(b, c, name="output")


    with tf.name_scope("update"):
        update_total = total_output.assign_add(output)
        increment_step = global_step.assign_add(1)

    with tf.name_scope("summary"):
        avg = tf.div(update_total, tf.cast(increment_step, tf.float32), name="average")

        tf.scalar_summary(b'Output', output, name="output_summary")
        tf.scalar_summary(b'sum of output over time', update_total, name="total_summary")
        tf.scalar_summary(b'average of output over time', avg, name="average_summary")



#============================chapter 4 =====================================
import tensorflow as tf 
def inference(X):
    pass

def loss(X, Y):
    pass

def inputs():
    pass

def train(total_loss):
    pass

def evaluate(sess, X, Y):
    pass


saver = tf.train.Saver()

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    X, Y = inputs()
    total_loss = loss(X, Y)
    train_op = train(total_loss)
    coord = tf.train.Coordinator()
    threads = tf.tranin_start_runner(sess=sess, coord=coord)

    initial_step = 0
    training_steps = 1000
    
    ckpt = tf.train.get_checkpoint_state(os.path.dirname=(__file__))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])


    for step in range(initial_step, training_steps):
        sess.run([train_op])

        if step % 1000 == 0:
            saver.save(sess, 'my-model', global_step=step)

    saver.save(sess, 'my-model', global_step=training_steps)
    sess.close()


