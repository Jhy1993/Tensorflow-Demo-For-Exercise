# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 20:49:20 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

RNNhttp://tensorfly.cn/tfdoc/tutorials/recurrent.html
"""

lstm = rnn_cell.BasicLSTMCell(lstm_size)
#initial LSTM storage state
state = tf.zeros([batch_size, lstm.state_size])

loss = 0.0
for current_batch_of_words in words_in_dataset:
    # update state after process a batch of words
    output, state = lstm(current_batch_of_words, state)
    #lstm predict the next word
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities = tf.nn.softmax(logits)
    loss += loss_function(probaiblities, target_words)

#input placeholder in given iteration, bptt on num_steps
words = tf.placeholder(tf.int32, [batch_size, num_steps])
lstm = rnn_cell.BasicLSTMCell(lstm_size)

initial_state = state = tf.zeros([batch_size, lstm.state_size])

for i in range(len(num_steps)):
    output, state = lstm(words[:, i], state)

final_state = state
# a numpy list save lstm state after every batch words
numpy_state = initial_state.eval()
total_loss = 0.0
for current_batch_of_words in words_in_dataset:
    numpy_state, current_loss = sess.run([final_state, loss],
                                        feed_dict={initial_state: numpy_state, words: current_batch_of_words})
    total_loss += current_loss

# embedding matrix : [vocabulary_size, embedding_size]
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)

#==========================================
lstm = rnn_cell.BasicLSTMCell(lstm_size)
stacked_lstm = rnn_cell.MultiRNNCell([lstm] * number_of_layers)

initial_state = state = stacked_lstm.zeros_state(batch_size, tf.float32)
for i in range(len(num_steps)):
    output, state = stacked_lstm(word[:, i], state)

final_state = state 
