import os
import sys
import time
import yaml

from python.DataLoaderBase import *
from python.DataLoader import DataLoader

with open(os.path.abspath( "configs/training_v1.yaml")) as f:
    config = yaml.safe_load(f)
scaling  = os.path.abspath("configs/ShuffleMergeSpectral_trainingSamples-2_files_0_50.json")
config["SetupNN"]["n_batches"] = 1000
config["Setup"]["n_tau"] = 1
dataloader = DataLoader(config, scaling)

gen_train = dataloader.get_generator(primary_set = True)
# gen_val = dataloader.get_generator(primary_set = False)

netConf_full = dataloader.get_net_config()
input_shape, input_types = dataloader.get_input_config()

data_train = tf.data.Dataset.from_generator(
    gen_train, output_types = input_types, output_shapes = input_shape
    ).prefetch(tf.data.AUTOTUNE)

time_checkpoints = [time.time()]
diff = []

for epoch in range(1):
    print("Epoch ->", epoch)
    for i,_ in enumerate(data_train):
        # if i % 10 == 0:
        #     time.sleep(10)
        time_checkpoints.append(time.time())
        diff.append(time_checkpoints[-1] - time_checkpoints[-2])
        print(i, " ", diff[-1], "s.")

del diff[0]
print("mean:", sum(diff)/len(diff))