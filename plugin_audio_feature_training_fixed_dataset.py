
import tensorflow as tf
import numpy as np
from tqdm import trange
import sys
sys.path.append('/home/tollie/Development/TensorFlowSynthProgrammers/utils/')

from plugin_feature_extractor import PluginFeatureExtractor
from tensorflow.contrib import rnn

extractor = PluginFeatureExtractor(midi_note=24, note_length_secs=0.8,
                                   render_length_secs=1.8,
                                   pickle_path="/home/tollie/Development/TensorFlowSynthProgrammers/utils/normalisers/",
                                   warning_mode="ignore")

path = "/home/tollie/Development/vsts/dexed/Builds/Linux/build/Dexed.so"
extractor.load_plugin(path)

if extractor.need_to_fit_normalisers():
    extractor.fit_normalisers(1)

(features, parameters) = extractor.get_random_normalised_example()

learning_rate = 0.001
training_iters = 400000
batch_size = 128
train_size = 20000
display_step = 100
save_step = 100
number_hidden = 100
number_layers = 3
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
    sp_x = tf.split(re_x, number_timesteps, 0)

    # embeddings = tf.get_variable('embedding_matrix', [number_outputs, number_hidden])
    # lstm_inputs = tf.nn.embedding_lookup(embeddings, x)
    lstm_cell = rnn.BasicLSTMCell(number_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = rnn.MultiRNNCell([lstm_cell] * number_layers, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, tf.float32)
    outputs, states = rnn.static_rnn(cell=lstm_cell, inputs=sp_x, dtype=tf.float32, initial_state=init_state)

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

with tf.name_scope("Cost"):
    cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, prediction))))
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    train_cost = tf.summary.scalar("Train Cost", cost)
    test_cost = tf.summary.scalar("Test Cost", cost)

# Initializing the variables.
init = tf.global_variables_initializer()

# Add ops to save and restore all variables.
saver = tf.train.Saver()

# Launching the graph.
with tf.Session() as sess:

    writer = tf.summary.FileWriter("./logs/deep_rnn_fixed_data_logs", sess.graph)
    merged = tf.summary.merge_all()

    # Restore variables from disk.
    # saver.restore(sess, "models/deep_rnn_model_fixed_data.ckpt")
    # print("Model restored.")
    sess.run(init)

    amount_datasets = 0

    while amount_datasets < 1:
        step = 1

        examples = []
        for i in trange(train_size, desc="Rendering examples"):
            examples += [extractor.get_random_normalised_example()]

        training_data = map(list, zip(*examples))
        batch_x = training_data[0]
        batch_y = training_data[1]

        (f, p) = extractor.get_random_normalised_example()
        f_shape = np.array(f).shape
        test_batch_x = np.zeros((batch_size, f_shape[0], f_shape[1]), dtype=np.float32)
        test_batch_y = np.zeros((batch_size, p.shape[0]), dtype=np.float32)

        for i in trange(batch_size, desc="Generating Test Batch"):
            (features, parameters) = extractor.get_random_normalised_example()
            test_batch_x[i] = features
            test_batch_y[i] = parameters

        train_step = 1
        while step * batch_size < training_iters:

            if (train_step * batch_size > train_size):
                train_step = 1

            start = (train_step - 1) * batch_size
            end = train_step * batch_size

            _, train_loss = sess.run([optimiser, train_cost], feed_dict={ x: batch_x[start:end],
                                                                          y: batch_y[start:end] })
            writer.add_summary(train_loss, (step - 1))

            if (step - 1) % display_step == 0:

                loss, test_loss = sess.run([cost, test_cost], feed_dict={ x: test_batch_x,
                                                                          y: test_batch_y })

                writer.add_summary(test_loss, (step - 1))

                print "    *** Model Status: " + "{:1f}".format(float(step * batch_size) / training_iters * 100.0) + \
                      "%" + " Finished. Test Batch Loss: " + \
                      "{:.9f}".format(loss) + " ***"

            if (step - 1) % save_step == 0:
                saved_path = saver.save(sess, "models/last_attempt_deep_rnn_model_fixed_data.ckpt")
                print "Model saved in file: " + saved_path
            train_step += 1
            step += 1
        amount_datasets += 1
print "Finished!"
