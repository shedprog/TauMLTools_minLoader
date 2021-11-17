import tensorflow as tf
import multiprocessing as mp
import numpy as np
import os
import yaml
import time

# def gen_wrap(n_puts):

#     def generator(n):
#         while True:
#             X = [
#             np.random.uniform(-1, 1, size = (250, 43)),
#             np.random.uniform(-1, 1, size = (250, 11, 11, 86)),
#             np.random.uniform(-1, 1, size = (250, 11, 11, 64)),
#             np.random.uniform(-1, 1, size = (250, 11, 11, 38)),
#             np.random.uniform(-1, 1, size = (250, 21, 21, 86)),
#             np.random.uniform(-1, 1, size = (250, 21, 21, 64)),
#             np.random.uniform(-1, 1, size = (250, 21, 21, 38)),
#             np.random.uniform(-1, 1, size = (250,)),
#             np.random.uniform(-1, 1, size = (250, 4)),
#             ]
#             X = [ tf.convert_to_tensor(e, dtype=tf.float32) for e in X ]
#             yield tuple(X)

#     return generator

# signature = tf.TensorSpec(shape=((None, 43),
#                             (None, 11, 11, 86),
#                             (None, 11, 11, 64),
#                             (None, 11, 11, 38),
#                             (None, 21, 21, 86),
#                             (None, 21, 21, 64),
#                             (None, 21, 21, 38),
#                             (None,), (250, 4),),
#                          dtype=(tf.float32,
#                             tf.float32,
#                             tf.float32,
#                             tf.float32,
#                             tf.float32,
#                             tf.float32,
#                             tf.float32,
#                             tf.float32,
#                             tf.float32,))



# dataset = tf.data.Dataset.range(1, 6) 
# dataset = dataset.interleave(
#     lambda x:
#         tf.data.Dataset.from_generator(gen_wrap(n_puts_),
#                                        args = (x,),
#                                        output_signature=signature),
#     cycle_length=6,
#     num_parallel_calls=6,
#     deterministic=False)

# for z in dataset:
#     print("get")

# n_puts_.value = 0

class ArtificialDataset(tf.data.Dataset):
    def _generator(num_samples):

        for sample_idx in range(num_samples):
            X = [
                np.random.uniform(-1, 1, size = (250, 43)),
                np.random.uniform(-1, 1, size = (250, 11, 11, 86)),
                np.random.uniform(-1, 1, size = (250, 11, 11, 64)),
                np.random.uniform(-1, 1, size = (250, 11, 11, 38)),
                np.random.uniform(-1, 1, size = (250, 21, 21, 86)),
                np.random.uniform(-1, 1, size = (250, 21, 21, 64)),
                np.random.uniform(-1, 1, size = (250, 21, 21, 38)),
                np.random.uniform(-1, 1, size = (250,)),
                np.random.uniform(-1, 1, size = (250, 4)),
            ]
            X = [ tf.convert_to_tensor(e, dtype=tf.float32) for e in X ]
            yield tuple(X)

    def __new__(cls, num_samples=10):
        shape=((None, 43),
                (None, 11, 11, 86),
                (None, 11, 11, 64),
                (None, 11, 11, 38),
                (None, 21, 21, 86),
                (None, 21, 21, 64),
                (None, 21, 21, 38),
                (None,), (250, 4),
               )
        dtype=(tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,
                tf.float32,)
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types = dtype, output_shapes = shape,
            args=(num_samples,)
        )

def benchmark(dataset, num_epochs=1):

    time_checkpoints = [time.time()]
    diff = []
    for epoch in range(num_epochs):
        print("Epoch ->", epoch)
        for i,_ in enumerate(dataset):
            # if i % 10 == 0:
            #     time.sleep(10)
            time_checkpoints.append(time.time())
            diff.append(time_checkpoints[-1] - time_checkpoints[-2])
            print(i, " ", diff[-1], "s.")
    del diff[0]
    print("mean:", sum(diff)/len(diff))

benchmark(
    tf.data.Dataset.range(10)
    .interleave(
        lambda _: ArtificialDataset(),
        num_parallel_calls=10
    )
)