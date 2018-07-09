import numpy as np
import random, time, pickle, math
import os, sys, glob

DIRECTORY = "outputs"
N = 10000000
ROUNDS = 8000
bars = " ▁▂▃▄▅▆▇█"
DOGS = True
VERBOSE = False

def fgen(p):
    "Return a function that returns log(exp(p*x) + exp(p*y))/p"
    if p == -np.inf:
        def f(x):
            return x.min(axis=1)
    elif p == np.inf:
        def f(x):
            return x.max(axis=1)
    elif p > 0:
        def f(x):
            # for numerical stability, we compute as the largest plus a correction term < log(2)/p
            a, b = x.min(axis=1), x.max(axis=1)
            return b + np.log(1 + np.exp(p*(a-b)))/p
    elif p < 0:
        def f(x):
            # for numerical stability, we compute as the smallest plus a correction term < log(2)/p
            a, b = x.min(axis=1), x.max(axis=1)
            return a + np.log(1 + np.exp(p*(b-a)))/p
    else:
        raise Exception("p = 0 is not implemented")
    return f

def integer_with_expectation(x):
    "Return an integer with expectation x"
    return int(x + random.random())

def recurse_on_samples(x, n, p0, f):
    # build n Samples
    # the first ~ n * p0 samples should be type 1
    # the second ~ n * p0 samples should be type 2
    n0 = integer_with_expectation(n * p0)
    n1 = n - n0
    result = np.zeros(n)
    # choose n pairs of samples from x
    x = np.random.choice(x, (n, 2))
    slice1 = slice(0, n0)
    slice2 = slice(n0, None)
    result[slice1] = f[0](x[slice1, :])
    result[slice2] = f[1](x[slice2, :])
    return result

def hist(x, range, n):
    return np.histogram(x, bins=n, range=range)

def generate_initial_samples():
    return np.full(N, 0.)

def printbars(h):
    # we already computed the histogram, so we don't want to recompute it
    # reduce the histogram with 9999 entries to a histogram with 33 entries
    h2 = np.reshape(h[0], (33, 303))
    h2 = h2.sum(axis=1)
    H = h2 / np.max(h2) * 8.99
    s = "".join([bars[int(x)] for x in H])
    print(s, end='')

def compute(p1, p2):
    f = [fgen(p1), fgen(p2)]
    start = time.time()
    def cur():
        return time.time() - start
    x = generate_initial_samples()
    print("Samples: %d" % N)
    tt = time.strftime("%Y-%m-%d-%H:%M:%S")
    d = DIRECTORY + "/%.2f,%.2f" % (p1, p2)
    if os.path.isdir(d):
        print("Output in existing directory: %s" % d)
    else:
        print("Trying to create directory: %s" %d)
        try:
            os.mkdir(d)
            print("Succeeded.")
            print("Output in new directory: %s" % d)
        except Exception as e:
            print("Could not make output directory %s" % d)
            print(e)
            return
    def timeestimates(i):
        c = cur()
        time_per_round = c / i
        time_left = (ROUNDS - i) * time_per_round
        total_time = c + time_left
        hour_minute = lambda x: (x / 3600, (x/60) % 60)
        print("Time estimates. Total: %02d:%02d. Remaining: %02d:%02d." %
            (hour_minute(total_time) + hour_minute(time_left)))
    for i in range(1,ROUNDS+1):
        # do recursions
        x = recurse_on_samples(x, N, 0.5, f)
        h = hist(x, None, 9999)
        if i % 120 == ROUNDS % 120:
            outputfile = d + "/output-%s-%06d" % (tt, i)
            output = open(outputfile, "wb")
            pickle.dump(h, output)
            output.close()
            if not VERBOSE:
                timeestimates(i)
                sys.stdout.flush()

        if VERBOSE:
            # update screen
            timeestimates(i)
            print(("Round %" + str(len(str(ROUNDS))) + "d/%d.") % (i, ROUNDS))
            s = str(np.random.choice(x, 24))
            lines = len(s.split("\n"))
            print(s)
            while lines < 4:
                print()
                lines += 1
            print("Histogram: ", end='')
            printbars(h)
            print("\r\033[A\r\033[A" + "\r\033[A" * lines, end='')
    if VERBOSE:
        for i in range(6):
            print()

def files_that_can_be_loaded():
    """Lists every file in the output directory.
    
    fl = combiner.files_that_can_be_loaded()
    All available choices of p, q:
    sorted(fl.keys())
    All available iteration counts for a specific p, q:
    sorted(fl[(p, q)].keys())
    All the runs for a specific p, q, iteration count:
    fl[(p, q)][iterations]

    You can load the files with combiner.loadfile
    """
    fl = {}
    def add(p1, p2, iterations, f):
        a = fl.setdefault((p1, p2), {})
        b = a.setdefault(iterations, [])
        b.append(f)
    for fn in glob.glob(DIRECTORY + "*,*/output-*-*"):
        d, f = fn.split("/")[-2:]
        p1, p2 = map(float, d.split(","))
        n = int(fn.split("-")[-1])
        add(p1, p2, n, fn)
    return fl

def loadfile(fn):
    """Load a histogram from filename. See files_that_can_be_loaded.
    There are 9999 bins.
    Format: a histogram of the log values like produced by numpy.histogram.
    Specifically, a tuple of two numpy arrays:
        array 1, samples in each bin (length 9999)
        array 2, separators between the bins (length 10000)
    
    To plot, you can say, for example,
    import matplotlib.pyplot as p
    h = combiner.loadfile(fn)
    samples, separators = h
    avgs = 0.5 * separators[:-1] + 0.5 * separators[1:]
    p.plot(samples, separators)
    p.show()"""
    fp = open(fn, "rb")
    h = pickle.load(fp)
    return h

def simpleplot(fn):
    "an example plot routine"
    import matplotlib.pyplot as p
    h = combiner.loadfile(fn)
    samples, separators = h
    samples = samples / np.sum(samples)
    samples = np.cumsum(samples)
    avgs = 0.5 * separators[:-1] + 0.5 * separators[1:]
    p.plot(avgs, samples)
    p.show()

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    import sys
    def exit(message):
        print(message, file=sys.stderr)
        sys.exit()
    try: p1, p2 = map(float, sys.argv[1:])
    except: exit("combiner <p1> <p2>")
    if DIRECTORY is None:
        exit("Open combiner.py and set the DIRECTORY to the output directory")
    compute(p1, p2)
