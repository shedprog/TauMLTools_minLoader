from multiprocessing import Process, Queue, Array
import multiprocessing as mp
import numpy as np
import time

def get_array():
    return np.concatenate((
            np.random.uniform(-1, 1, size = (250, 43)).flatten(),
            np.random.uniform(-1, 1, size = (250, 11, 11, 86)).flatten(),
            np.random.uniform(-1, 1, size = (250, 11, 11, 64)).flatten(),
            np.random.uniform(-1, 1, size = (250, 11, 11, 38)).flatten(),
            np.random.uniform(-1, 1, size = (250, 21, 21, 86)).flatten(),
            np.random.uniform(-1, 1, size = (250, 21, 21, 64)).flatten(),
            np.random.uniform(-1, 1, size = (250, 21, 21, 38)).flatten(),
            np.random.uniform(-1, 1, size = (250,)).flatten(),
            np.random.uniform(-1, 1, size = (250, 4)).flatten(),
        ), axis=0)

class ArrayQueue(object):
    def __init__(self, template, maxsize=0):
        if type(template) is not np.ndarray:
            raise ValueError('ArrayQueue(template, maxsize) must use a numpy.ndarray as the template.')
        if maxsize == 0:
            # this queue cannot be infinite, because it will be backed by real objects
            raise ValueError('ArrayQueue(template, maxsize) must use a finite value for maxsize.')

        # find the size and data type for the arrays
        # note: every ndarray put on the queue must be this size
        self.dtype = template.dtype
        self.shape = template.shape
        self.byte_count = len(template.data)

        # make a pool of numpy arrays, each backed by shared memory, 
        # and create a queue to keep track of which ones are free
        self.array_pool = [None] * maxsize
        self.free_arrays = Queue(maxsize)
        for i in range(maxsize):
            buf = Array('d', self.byte_count, lock=False)
            self.array_pool[i] = np.frombuffer(buf, dtype=self.dtype).reshape(self.shape)
            self.free_arrays.put(i)

        self.q = Queue(maxsize)

    def put(self, item, *args, **kwargs):
        if type(item) is np.ndarray:
            if item.dtype == self.dtype and item.shape == self.shape and len(item.data)==self.byte_count:
                # get the ID of an available shared-memory array
                id = self.free_arrays.get()
                # copy item to the shared-memory array
                self.array_pool[id][:] = item
                # put the array's id (not the whole array) onto the queue
                new_item = id
            else:
                raise ValueError(
                    'ndarray does not match type or shape of template used to initialize ArrayQueue'
                )
        else:
            # not an ndarray
            # put the original item on the queue (as a tuple, so we know it's not an ID)
            new_item = (item,)
        self.q.put(new_item, *args, **kwargs)

    def get(self, *args, **kwargs):
        item = self.q.get(*args, **kwargs)
        if type(item) is tuple:
            # unpack the original item
            return item[0]
        else:
            # item is the id of a shared-memory array
            # copy the array
            arr = self.array_pool[item].copy()
            # put the shared-memory array back into the pool
            self.free_arrays.put(item)
            return arr

def writer(output):


    while True:
        # X = [
        #     np.random.uniform(-1, 1, size = (250, 43)),
        #     np.random.uniform(-1, 1, size = (250, 11, 11, 86)),
        #     np.random.uniform(-1, 1, size = (250, 11, 11, 64)),
        #     np.random.uniform(-1, 1, size = (250, 11, 11, 38)),
        #     np.random.uniform(-1, 1, size = (250, 21, 21, 86)),
        #     np.random.uniform(-1, 1, size = (250, 21, 21, 64)),
        #     np.random.uniform(-1, 1, size = (250, 21, 21, 38)),
        #     np.random.uniform(-1, 1, size = (250,)),
        #     np.random.uniform(-1, 1, size = (250, 4)),
        # ]
            # X = [
        # print(X.shape)
        # ]


        X = get_array()
        output.put(X)

def gen():
    N = 10
    # output = mp.Queue(20)
    X = get_array()
    dtype = X.dtype
    shape = X.shape
    byte_count = len(X.data)
    print(dtype, shape, byte_count)
    output = ArrayQueue(template=X, maxsize=30)
    processes = []
    for i in range(N):
        processes.append(
        mp.Process(target = writer, args = (output,)))
        processes[-1].deamon = True
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
        if i>100:
            break

del diff[0]
print("mean:", sum(diff)/len(diff))

# class __EndToken(object):
#     pass

# def parallel_pipeline(buffer_size=50):
#     def parallel_pipeline_with_args(f):
#         def consumer(xs, q):
#             for x in xs:
#                 q.put(x)
#             q.put(__EndToken())

#         def parallel_generator(f_xs):
#             q = ArrayQueue(template=np.zeros(0,1,(500,2000)), maxsize=buffer_size)
#             consumer_process = Process(target=consumer,args=(f_xs,q,))
#             consumer_process.start()
#             while True:
#                 x = q.get()
#                 if isinstance(x, __EndToken):
#                     break
#                 yield x

#         def f_wrapper(xs):
#             return parallel_generator(f(xs))

#         return f_wrapper
#     return parallel_pipeline_with_args


# @parallel_pipeline(3)
# def f(xs):
#     for x in xs:
#         yield x + 1.0

# @parallel_pipeline(3)
# def g(xs):
#     for x in xs:
#         yield x * 3

# @parallel_pipeline(3)
# def h(xs):
#     for x in xs:
#         yield x * x

# def xs():
#     for i in range(1000):
#         yield np.random.uniform(0,1,(500,2000))

# print "multiprocessing with shared-memory arrays:"
# %time print sum(r.sum() for r in f(g(h(xs()))))   # 13.5s


# Scaling results:
# 1 - 0.782s
# 2 - 0.333s
# 4 - 0.182s
# 6 - 0.162s
# 8 - 0.159s