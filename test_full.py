import os
import sys
import time

from ..python.DataLoader import DataLoader

config   = os.path.abspath( "../../configs/training_v1.yaml")
scaling  = os.path.abspath("../../configs/scaling_params_v1.json")
dataloader = DataLoader.DataLoader(config, scaling)

gen_train = dataloader.get_generator(primary_set = True)

netConf_full, input_shape, input_types  = dataloader.get_config()

data_train = tf.data.Dataset.from_generator(
    gen_train, output_types = input_types, output_shapes = input_shape
    ).prefetch(10)

time_checkpoints = [time.time()]

for epoch in range(3):
    print("Epoch ->", epoch)
    for i,_ in enumerate(data_train):
        time_checkpoints.append(time.time())
        print(i, " ", time_checkpoints[-1]-time_checkpoints[-2], "s.")


