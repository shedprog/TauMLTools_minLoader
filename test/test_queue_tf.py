import multiprocessing as mp
import tensorflow as tf
import numpy as np
import time

def writer(output):


    while True:
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
        # X = [ tf.convert_to_tensor(e, dtype=tf.float32) for e in X ]
        print("try")
        output.enqueue(X)
        print("done")
        break

def gen():
    N = 5
    # output = mp.Queue(20)
    output = tf.queue.FIFOQueue(20, [tf.float32,
                          tf.float32,
                          tf.float32,
                          tf.float32,
                          tf.float32,
                          tf.float32,
                          tf.float32,
                          tf.float32,
                          tf.float32,],
                    shapes=((250, 43),
                            (250, 11, 11, 86),
                            (250, 11, 11, 64),
                            (250, 11, 11, 38),
                            (250, 21, 21, 86),
                            (250, 21, 21, 64),
                            (250, 21, 21, 38),
                            (250,), (250, 4),
                            ))
    processes = []
    for i in range(N):
        processes.append(
        mp.Process(target = writer, args = (output,)))
        processes[-1].deamon = True
        processes[-1].start()

    while True:
        print(output.size())
        yield 0
        # print("try get")
        # yield output.dequeue()
        # print("done get")

    for i, pr in enumerate(processes):
        pr.join()

if __name__=='__main__':

    time_checkpoints = [time.time()]
    diff = []

    for i, x in enumerate(gen()):

        time_checkpoints.append(time.time())
        diff.append(time_checkpoints[-1] - time_checkpoints[-2])
        print(i, " ", diff[-1], "s.")
        if i>200:
            break

del diff[0]
print("mean:", sum(diff)/len(diff))


# Tests:
# 1  workers - mean 0.596 s
# 2  workers - mean 0.370 s
# 3  workers - mean 0.451 s
# 5  workers - mean 0.372 s
# 10 workers - mean 0.484 s
