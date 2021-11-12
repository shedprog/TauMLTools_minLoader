import torch.multiprocessing as mp
import torch
import time
import numpy as np

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
        X = [ torch.tensor(e, dtype=torch.float32) for e in X ]
        output.put(X)

def gen():
    N = 5
    output = mp.Queue(20)
    # output = fmq.Queue(maxsize=20)
    processes = []
    for i in range(N):
        processes.append(
        mp.Process(target = writer, args = (output,)))
        # processes[-1].deamon = True
        processes[-1].start()

    while True:
        yield output.get()

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