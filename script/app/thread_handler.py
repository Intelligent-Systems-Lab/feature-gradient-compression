import time
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool
import threading
import copy

def pp(val, val2):
    print("Hi, {}, {}".format(val, val2))
    return val


class Treading:
    def __init__(self, proc=5):
        # self.pool = ThreadPool(proc)
        self.proc = proc

    def run(self, fs=None, round_=None, model=None):
        if (fs is None) or (round_ is None) or (model is None):
            return

        results = []
        p = Pool(self.proc)
        for cid, f in zip(range(len(fs)), fs):
            results.append(p.apply_async(f.train_run, (cid, copy.deepcopy(model),)))
            time.sleep(0.5)
        p.close()
        p.join()
        results = [r.get() for r in results]
        return results
        #
        # results = []
        # for cid, f in zip(range(len(fs)), fs):
        #     results.append(self.pool.starmap_async(f.train_run, [(cid, copy.deepcopy(model))]))
        #
        # while True:
        #     l = [p.is_alive() for p in results]
        #     print(l)
        #     if sum(l) == 0:
        #         break
        #     time.sleep(3)
        #
        #
        # self.pool.close()
        # self.pool.join()
        # results = [r.get()[0] for r in results]
        # return results


def create_job(worker, args=()) -> threading.Thread:
    t = threading.Thread(name="new th", target=worker, args=args)
    # t.start()
    return t
