import threading


def create_job(worker, args=()) -> threading.Thread:
    t = threading.Thread(name="new th", target=worker, args=args)
    # t.start()
    return t
