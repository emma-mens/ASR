import sys
import numpy as np
import tensorflow as tf
import argparse
from utils import GetData
import matplotlib.pyplot as plt

NUM_FREQ_BINS = 1025
SAMPLES_PER_FRAME = 20
SAVE_ITERATIONS = 20
CONFIDENCE = 0.5
GPU_ON = True

n_input_units = NUM_FREQ_BINS * SAMPLES_PER_FRAME
n_hidden_units_h0 = NUM_FREQ_BINS * SAMPLES_PER_FRAME
n_output_units = NUM_FREQ_BINS * SAMPLES_PER_FRAME
fetch_data = GetData()
batch_size = 10
n_channels = 1
dropout = 0.0# No dropout
test_pred_conf = 0.5 # threshold for prediction on test

if len(sys.argv) == 2:
  fetch_data = GetData(sys.argv[1])

X = tf.placeholder(tf.float32, shape=(None, n_input_units), name='data_placeholder')

y = tf.placeholder(tf.float32, shape=(None, n_output_units), name='labels_placeholder')

keep_prob = tf.placeholder(tf.float32)

test_pred_confidence = tf.placeholder(tf.float32)

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=.01), name='w_conv1'),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=.01), name='w_conv2'),
    # fully connected, 257*5*64 inputs (two downsampling of input), 1024 outputs
    'wd1': tf.Variable(tf.truncated_normal([257*5*64, 1024], stddev=.01), name='w_dense1'),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([1024, n_output_units], stddev=.01), name='w_dense2')
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([32], stddev=.01), 'b_conv1'),
    'bc2': tf.Variable(tf.truncated_normal([64], stddev=.01), 'b_conv2'),
    'bd1': tf.Variable(tf.truncated_normal([1024], stddev=.01), 'b_dense1'),
    'out': tf.Variable(tf.truncated_normal([n_output_units], stddev=.01), 'b_dense2')
}

parser = argparse.ArgumentParser(description='ASR Training.')
parser.add_argument('--checkpoint', type=str, dest='checkpoint_path',
                    help="checkpoint file to save model", required=True)

parser.add_argument('--save-iterations', type=int, dest='save_iterations',
                    help="number of iterations between successive model saves",
                    default=SAVE_ITERATIONS)

parser.add_argument('--confidence', type=int, dest='confidence', 
                    help="threshold for predicting a network output bin as vocal",
                    default=CONFIDENCE)

parser.add_argument('--verbose', action='store_true', default=False)

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture. shape=[infered_batch_size, height, width, n_channels]
    x = tf.reshape(x, shape=[-1, NUM_FREQ_BINS, SAMPLES_PER_FRAME, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1, name='fully_connected')
    
    # Apply Dropout
    # fc1 = tf.nn.dropout(fc1, dropout)

    # Output
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    out = tf.sigmoid(out, name='sigmoid_output')
    return out

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME', use_cudnn_on_gpu=GPU_ON)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

with tf.device('/gpu:0'):

    # Construct model
    out = conv_net(X, weights, biases, keep_prob)
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=y))
    learning_rate = 0.3 #TODO: experiment
    
    # Define the optimizer operation.  This is what will take the derivate of the loss 
    # with respect to each of our parameters and try to minimize it.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    out = tf.sigmoid(out, name='sigmoid_output')
    prediction = tf.cast(tf.greater(out, confidence), tf.float32) #TODO: experiment with confidence
    
    # Compute the accuracy
    prediction_is_correct = tf.equal(prediction, y)
    accuracy = tf.reduce_mean(tf.cast(prediction_is_correct, tf.float32))

def shape_data(d):
    # Converts 2d arrays in the batch to 1D arrays in the batch
    s = np.shape(d)
    return np.reshape(d, (s[0], s[1]*s[2]))

t_accuracy = []
t_loss = []
saver = tf.train.Saver()

if __name__ == "__main__":
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:

        # this operation initializes all the variables we made earlier.
        tf.global_variables_initializer().run()

        for epoch in xrange(20):
        
            sub_t_loss = []
            sub_t_accuracy = []
            sub_test_accuracy = []
            fetch_data.next_song_idx = 0

            for step in xrange(fetch_data.NUM_TRAIN_SONGS):

                full_x, full_y = fetch_data.get_next_train_data()

                feed_dict = {X: shape_data(full_x[:full_x.size/3,:]), 
                                y: shape_data(full_y[:full_y.size/3,:]), 
                                keep_prob:dropout, pred_size:1}

                _, loss_value_train, predictions_value_train, accuracy_value_train = session.run(
                    [optimizer, loss, prediction, accuracy], feed_dict=feed_dict)
                
                if verbose:
                    #TODO: control logging (GLOG?)
                    print "epoch", epoch, "step", step, "prediction accuracy:", accuracy_value_train, "loss", loss_value_train
                
                #get trends for graphing (#TODO: use tensorboard)
                sub_t_loss.append(loss_value_train)
                sub_t_accuracy.append(accuracy_value_train)
            t_accuracy.append(sub_t_accuracy)
            t_loss.append(sub_t_loss)
            
            if epoch % save_iterations == 0: 
                #TODO: add validation
                save_path = saver.save(session, checkpoint_path)
                if verbose:
                    print "Model saved in file: %s" % save_path    
            
    #TODO: remove after using tensorboard
    np.savetxt('train_accuracy_' + checkpoint_path, t_accuracy)
    np.savetxt('train_loss' + checkpoint_path, t_loss)
