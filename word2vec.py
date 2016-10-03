# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 21:01:47 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

http://tensorfly.cn/tfdoc/tutorials/word2vec.html
word2vec
"""

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))   
# 建立输入占位符
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
# 对批数据中的单词建立嵌套向量，
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
# 计算 NCE 损失函数, 每次使用负标签的样本.
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, num_sampled, vocabulary_size))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0).minimize(loss)
#训练的过程很简单，只要在循环中使用feed_dict不断给占位符填充数据，同时调用 session.run即可。
for inputs, labels in generate_batch(...):
    feed_dict = {training_inputs: inputs, training_labels: labels}
    _, cur_loss= session.run([optimizer, loss], feed_dict=feed_dict)
    