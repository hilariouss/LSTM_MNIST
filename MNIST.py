import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 1. Define graph

lr = 0.001              # learning rate
training_iters = 100000 # Number of training
batch_size = 128        # batch size
n_inputs = 28          # MNIST data input (img shape : 28 * 28)
n_steps = 28
n_hidden_units = 128    # Number of neurons in hidden layer of a LSTM cell
n_classes = 10          # MNIST classes (0-9 digits) : 0, 1, ..., 9

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])   # input : 28 x 28
y = tf.placeholder(tf.float32, [None, n_classes])           # class

weights = {'in':tf.Variable(tf.random_normal([n_inputs, n_hidden_units])), # 28, 128
           'out':tf.Variable(tf.random_normal([n_hidden_units, n_classes]))}
biases = {'in':tf.Variable(tf.constant(0.1, shape=[n_hidden_units])), # (1x 128)
          'out':tf.Variable(tf.constant(0.1, shape=[n_classes]))}     # ()

# 2. Defining the RNN

def RNN(X, weights, biases):
    # hidden layer for input to lstm cell

    # transpose the inputs shape (batch, n_steps, n_inputs) 3D --> (batch * n_steps, n_inputs) 2D
    X = tf.reshape(X, [-1, n_inputs])

    # flow from input layer to hidden layer --> matmul X and weights + biases
    # (128*28, 28) * (28, 128) + biases (128, )
    X_in = tf.matmul(X, weights['in'])
    print(X_in)
    X_in =  X_in + biases['in'] # (batch * n_steps, n_inputs) * (n_inputs, n_hidden_units)
    print(biases['in'])
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units]) # (batch, n_steps, n_hidden_units)

    LSTMcell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias = 1.0, state_is_tuple=True)
    init_state = LSTMcell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(LSTMcell, X_in, initial_state=init_state, time_major=False)

    #hidden layer for output as the final results
    #results = tf.matmul(final_state[1], weights['out']) + biases['out']
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results

# 3. Training and evaluation

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred,labels= y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

step = 0
while step * batch_size < training_iters:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
    inp = {x: batch_xs, y: batch_ys}
    sess.run([train_op], feed_dict=inp)
    if step % 20 == 0:
        inp = {x: batch_xs, y: batch_ys}
        print(sess.run(accuracy, feed_dict=inp))
    step += 1