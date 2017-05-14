from utils import PluginFeatureExtractor
import tensorflow as tf
import numpy as np
from tqdm import trange
from tensorflow.python.ops import rnn, rnn_cell

extractor = PluginFeatureExtractor(midi_note=24, note_length_secs=0.8,
                                   render_length_secs=1.8,
                                   pickle_path="normalisers/",
                                   warning_mode="ignore")

path = "/home/tollie/Development/vsts/dexed/Builds/Linux/build/Dexed.so"
extractor.load_plugin(path)

if extractor.need_to_fit_normalisers():
    extractor.fit_normalisers(10000)

(features, parameters) = extractor.get_random_normalised_example()

learning_rate = 0.001
training_iters = 2000000
batch_size = 1
display_step = 10
save_step = 50
number_hidden = 20
number_layers = 16
number_input = int(features.shape[1])
number_timesteps = int(features.shape[0])
number_outputs = len(parameters)

x = tf.placeholder("float", [None, number_timesteps, number_input], name="Features")
y = tf.placeholder("float", [None, number_outputs], name="Synth_Patch")

# Create model
def RNN(x, weights, biases):
    # Prepare data shape to match rnn function requirements.
    # Current data input shape: (batch_size, number_timesteps, number_input)
    # Required shape: (number_timesteps, batch_size, number_input)
    tr_x = tf.transpose(x, [1, 0, 2])
    re_x = tf.reshape(tr_x, [-1, number_input])
    sp_x = tf.split(0, number_timesteps, re_x)

    # embeddings = tf.get_variable('embedding_matrix', [number_outputs, number_hidden])
    # lstm_inputs = tf.nn.embedding_lookup(embeddings, x)

    lstm_cell = rnn_cell.LSTMCell(number_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = rnn_cell.MultiRNNCell([lstm_cell] * number_layers, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, tf.float32)
    outputs, states = rnn.rnn(cell=lstm_cell, inputs=sp_x, dtype=tf.float32, initial_state=init_state)

    # Linear activation using rnn inner loop last output
    return tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'])

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

# Store layers weight & bias
weights = {
    'out': init_weights([number_hidden, number_outputs], "weights_out")
}
biases = {
    'out': init_weights([number_outputs], "biases_out")
}

tf.summary.histogram("out", weights['out'])

# Construct model
prediction = RNN(x, weights, biases)

# Add ops to save and restore all variables.
saver = tf.train.Saver()

(f, p) = extractor.get_random_normalised_example()
f_shape = np.array(f).shape

batch_x = np.zeros((batch_size, f_shape[0], f_shape[1]), dtype=np.float32)
batch_y = np.zeros((batch_size, p.shape[0]), dtype=np.float32)

# Launching the graph.
with tf.Session() as sess:

    # Restore variables from disk.
    saver.restore(sess, "models/deep_rnn_model.ckpt")
    print("Model restored.")

    for i in trange(batch_size, desc="Generating Batch"):
        (features, parameters) = extractor.get_random_normalised_example()
        batch_x[i] = features
        batch_y[i] = parameters

    print "Test model..."

    pred = sess.run([prediction], feed_dict={ x: batch_x,
                                              y: batch_y })
    print len(pred)
    print len(pred[0])
    print pred[0]
    all_tests = []

    for i in trange(len(pred[0]), desc="Creating results table"):
        total_abs = 0

        for param in range(len(pred[0][0])):
            pred_param = pred[0][i][param]
            actual_param = batch_y[i][param]
            # print actual_param
            total_abs += abs(pred_param - actual_param)

        absolute_distance_table = " " +("%04d" % i) + " : " + ("%.5f" % round(total_abs, 5))
        all_tests.append((total_abs, absolute_distance_table, pred[0][i], batch_y[i]))

    all_tests.sort(key=lambda x: x[0])

    print "\n\n      In order of most to least similar predictions compared to the actual parameters:\n\n      Index   Total abs"
    print "       _______________"
    for i in range(len(all_tests)):
        print ("%04d" % i) + ") " + all_tests[i][1]

    def add_patch_indices(patch):
        tuple_patch = []
        for i in range(len(patch)):
            tuple_patch += [(i, float(patch[i]))]
        return tuple_patch


    for i in range(len(all_tests)):
        patch_predicted = add_patch_indices(all_tests[i][2])
        patch_actual = add_patch_indices(all_tests[i][3])

        extractor.set_patch(patch_predicted)
        file_name = str(i) + "_pred.wav"
        extractor.write_to_wav(file_name)

        extractor.set_patch(patch_actual)
        file_name = str(i) + "_actual.wav"
        extractor.write_to_wav(file_name)
