import numpy as np
from utils import PluginFeatureExtractor
from tqdm import trange

print "Initialising state."
amount = 50000
file_name = 'dexed_examples.npy'
extractor = PluginFeatureExtractor(midi_note=24, note_length_secs=0.4,
                                   desired_features=[i for i in range(8, 21)],
                                   render_length_secs=0.7,
                                   pickle_path="utils/normalisers",
                                   warning_mode="ignore", normalise_audio=False)

path = "/home/tollie/Development/vsts/dexed/Builds/Linux/build/Dexed.so"
extractor.load_plugin(path)

if extractor.need_to_fit_normalisers():

    print "No normalisers found, fitting new ones."
    extractor.fit_normalisers(10000)


examples = []
for i in trange(amount, desc="Rendering examples"):
    examples += [extractor.get_random_normalised_example()]

np_examples = np.array(examples)
np.save(file_name, np_examples)

print "Finished."
