import argparse
import tensorflow as tf

#TODO: consider making these shared between files that need them
NUM_FREQ_BINS = 1025
SAMPLES_PER_FRAME = 20
CONFIDENCE = 0.5
n_output_units = NUM_FREQ_BINS * SAMPLES_PER_FRAME

fetch_data = GetData()
x_test, y_test = fetch_data.get_test_data()

def make_predictions(net_output, alpha):
    # Makes prediction for each frequency bin by averaging all predictions for that
    # bin over SAMPLES_PER_FRAME context predictions
    # net_output is the network prediction for all test frames at frameshift 1
    # alpha is the prediction threshold
    # TODO: optimize the prediction method through vectorization

    # Stores a buffer of SAMPLES_PER_FRAME for every frame index
    store = np.zeros((n_output_units, SAMPLES_PER_FRAME)) # Stores a buffer of 20 distribution for all frame indices
    predicted = np.zeros(net_output.shape[:2]) # all predicted binary masks for frame,freq bins

    for frame in xrange(predicted.shape[0]):

        prediction_values = net_output[frame].flatten()
        norm = float(min(frame, SAMPLES_PER_FRAME)) # first few frames won't have entire distribution

        # Zeroth index has entire distribution predicted so we can get the average prediction
        predicted[frame, :] = np.greater(np.add.reduce(np.reshape(store[:,0], (SAMPLES_PER_FRAME,NUM_FREQ_BINS)),\
                              axis=0)/(norm), alpha).astype(int)

        # Update store by pushing most recent one out and adding newest prediction
        store[:, 0:-1] = store[:, 1:store.shape[1]]

        # Update store content by adding prediction to corresponding frames
        for i in xrange(store.shape[1]):
            store[NUM_FREQ_BINS*(SAMPLES_PER_FRAME-i-1):NUM_FREQ_BINS*(SAMPLES_PER_FRAME-i), i] = \
                                              prediction_values[NUM_FREQ_BINS*i:NUM_FREQ_BINS*(i+1)] 
            
    return predicted

parser = argparse.ArgumentParser(description='ASR Evaluation.')

parser.add_argument('--checkpoint', type=str, dest='checkpoint',
                    help='checkpoint of model saved', required=True)

parser.add_argument('--confidence', type=int, dest='confidence', 
                    help="threshold for predicting a network output bin as vocal",
                    default=CONFIDENCE)

if __name__ == "__main__":
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    saver.restore(session, checkpoint)

    net_output = np.empty((x_test.shape[0], NUM_FREQ_BINS, SAMPLES_PER_FRAME))
    for frame in xrange(x_test.shape[0]):
        feed_dict_test = {X: shape_data(np.array([x_test[frame]])), y: shape_data(np.array([y_test[frame]])), keep_prob:dropout}
        loss_value_test, predictions_value_test = session.run([loss, out], feed_dict=feed_dict_test) 
        net_output[frame,:,:] = np.reshape(predictions_value_test, (1025,20))

    learned_mask = make_predictions(net_output, confidence)
    np.savetxt('test_accuracy', learned_mask) #TODO: save split spectrogram instead