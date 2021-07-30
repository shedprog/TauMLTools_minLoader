import tensorflow as tf
import multiprocessing as mp
import os
import yaml
import time

# def generator(n):
#     while True:
#         yield n

# def dataset(n):
#     return tf.data.Dataset.from_generator(generator(n))

# ds = tf.data.Dataset.range(10).apply(tf.contrib.data.parallel_interleave(dataset, cycle_lenght=10))

# for i in ds:
#     print(i)
# where N is the number of generators you use

# Preprocess 4 files concurrently, and interleave blocks of 16 records
# from each file.

# dataset = tf.data.Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]

# # NOTE: New lines indicate "block" boundaries.

# dataset = dataset.interleave(
#     lambda x: tf.data.Dataset.from_tensors(x).repeat(6),
#     cycle_length=2, block_length=4)

# print(list(dataset.as_numpy_iterator()))

n_puts_ = mp.Value('i', 0)

def gen_wrap(n_puts):
    def generator(n):
        while True:
            n_puts.value += 1
            yield n, n_puts.value
            if n_puts.value > 100:
                break
    return generator

dataset = tf.data.Dataset.range(1, 6) 
dataset = dataset.interleave(lambda x: tf.data.Dataset.from_generator(gen_wrap(n_puts_),
                                                                      args = (x,),
                                                                      output_signature=(tf.TensorSpec(shape=(2,), dtype=tf.int32))
                                                                     ),
cycle_length=6, num_parallel_calls=6, deterministic=False)

for i,z in dataset:
    print(i,z)

n_puts_.value = 0

for i,z in dataset:
    print(i,z)