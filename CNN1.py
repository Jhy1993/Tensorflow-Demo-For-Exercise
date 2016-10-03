def conv2d(x, w, b):
    # num * chanel * height * weight
    #height_conved = (height + 2* pad - kernal) / stride + 1
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='same'), b))
def max_pool_kxk(x):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], 
                        strides=[1, 1, 1, 1],
                        padding='SAME')

    
#input : w*h picture
x = tf.placeholder(tf.float32, [None, w*h])
y = tf.placeholder(tf.float32, [None, ysize])

wc = tf.Variables(tf.random_normal([3, 3, 1, 64]))
wc=tf.Variable(tf.random_normal([3, 3, 1, 64]) 
'''
3 3 分别为3x3大小的卷积核 1位输入数目 
因为是第一层所以是1 输出我们配置的64 
所以我们知道了 如果下一次卷积wc2=[5,5,64,256] //5x5 是我们配置的卷积核大小，
第三位表示输入数目 我们通过上面知道 上面的输出 也就是下一层的输入 所以 就是64 了
输出我们定义成256 
这个随你喜好，关键要迭代看效果，一般都是某一个v*v的值
'''
b1 = tf.Variable(tf.random_normal([64]))


c1 = tf.nn.relu(tf.nn.conv2d(X, w, [1, 1, 1, 1], 'SAME'))
m1 = tf.nn.max_pool()




