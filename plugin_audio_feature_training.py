from utils import PluginFeatureExtractor
import tensorflow as tf
import numpy as np
from tqdm import trange
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.framework import function

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
batch_size = 50
display_step = 1
save_step = 20
number_hidden = 40
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
def resize_grad(op, grad):
    x = op.inputs[0]
    return grad * x

@function.Defun(python_grad_func=resize_grad)
def resize_op(tensor):
    reshaped_tensor = tf.reshape(tensor, [-1, 31, 5])
    reshaped_tensor = tf.pad(reshaped_tensor,
                             [[0, 0], [20, 20], [8, 8]], "CONSTANT")
    return reshaped_tensor

prediction = RNN(x, weights, biases)
prediction = tf.identity(prediction)
prediction_reshaped_y_to_x = resize_op(prediction)
predicted_features = tf.identity(prediction_reshaped_y_to_x)

with tf.name_scope("Cost"):
    cost = tf.reduce_mean(tf.abs(tf.sub(x, predicted_features)))
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    tf.summary.scalar("Cost", cost)

# Initializing the variables.
init = tf.global_variables_initializer()

# Add ops to save and restore all variables.
saver = tf.train.Saver()

def add_patch_indices(patch):
    tuple_patch = []
    for i in range(len(patch)):
        tuple_patch += [(i, float(patch[i]))]
    return tuple_patch

# Launching the graph.
with tf.Session() as sess:

    writer = tf.summary.FileWriter("./logs/deep_rnn_logs", sess.graph)
    merged = tf.summary.merge_all()

    sess.run(init)
    step = 1

    (f, p) = extractor.get_random_normalised_example()
    f_shape = np.array(f).shape

    batch_x = np.zeros((batch_size, f_shape[0], f_shape[1]), dtype=np.float32)
    batch_y = np.zeros((batch_size, p.shape[0]), dtype=np.float32)

    while step * batch_size < training_iters:

        for i in trange(batch_size, desc="Generating Batch             "):
            (features, parameters) = extractor.get_random_normalised_example()
            batch_x[i] = features
            batch_y[i] = parameters

        print "Training model..."

        predictions = sess.run(prediction, feed_dict={ x: batch_x,
                                                       y: batch_y })
        features_from_predictions = []
        for i in trange(len(predictions), desc="Generating Predicted Features"):
            pred_feat = extractor.get_features_from_patch(add_patch_indices(predictions[i]))
            features_from_predictions += [pred_feat]
        features_from_predictions = np.array(features_from_predictions)
        sess.run(optimiser, feed_dict={ x: batch_x,
                                        y: batch_y,
                                        predicted_features: features_from_predictions })

        # summary = sess.run(merged, feed_dict={ x: batch_x,
        #                                        y: batch_y,
        #                                        predicted_features: features_from_predictions })
        #
        # writer.add_summary(summary, (step - 1))

        if (step - 1) % display_step == 0:
            for i in trange(batch_size, desc="Generating Test Batch        "):
                (features, parameters) = extractor.get_random_normalised_example()
                batch_x[i] = features
                batch_y[i] = parameters

            predictions = sess.run(prediction, feed_dict={ x: batch_x,
                                                           y: batch_y })
            features_from_predictions = []
            for i in trange(len(predictions), desc="Testing Predicted Features   "):
                pred_feat = extractor.get_features_from_patch(add_patch_indices(predictions[i]))
                features_from_predictions += [pred_feat]
            features_from_predictions = np.array(features_from_predictions)

            loss = sess.run(cost, feed_dict={ x: batch_x,
                                              y: batch_y,
                                              predicted_features: features_from_predictions })

            print "    *** Model Status: " + "{:1f}".format(float(step * batch_size) / training_iters * 100.0) + \
                  "%" + " Finished. Test Batch Loss: " + \
                  "{:.9f}".format(loss) + " ***\n"

        if (step - 1) % save_step == 0:
            saved_path = saver.save(sess, "models/deep_rnn_model.ckpt")
            print "Model saved in file: " + saved_path

        step += 1
print "Finished!"
