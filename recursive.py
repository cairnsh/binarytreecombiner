import numpy as np
import matplotlib.pyplot
import math

"""A module that can apply a function to two independent distributions
and calculate the resulting distribution.

Contact Hannah Cairns <ahc238@cornell.edu>"""

ESTIMATE = ["over", "under"]

class DistributionFunction:
    """DistributionFunction(epsilon, n0, n1, estimate=None, copyfrom=None)

    Holds a vector that is an overestimate or underestimate of a distribution function.
    The length of the vector is n1 - n0 + 2.
    The distance between successive points is epsilon.
    The i-th entry in the vector represents the value at epsilon * (i + n0 - 1).
    
    Estimate has to be "over" or "under".
    If you want to copy from another vector, you can use copyfrom."""
    def __init__(self, epsilon, n0, n1, estimate=None, copyfrom=None):
        self.epsilon = epsilon
        assert n0 <= n1
        self.zero = 1-n0
        shape = (n1 - n0 + 2,)
        if copyfrom is None:
            self.interval = np.zeros(shape)
            self.interval[0] = 0
            self.interval[-1] = 1
        else:
            assert copyfrom.shape == shape
            self.interval = np.copy(copyfrom)
        assert self.interval[-1] == 1
        assert estimate in ESTIMATE
        self.estimate = estimate
    def range(self):
        "Return the x coordinates corresponding to each entry."
        n = self.interval.shape[0]
        return (np.arange(n) - self.zero)*self.epsilon
    def params(self):
        "Return the parameters passed."
        n0 = 1-self.zero
        n1 = self.interval.shape[0] + n0 - 2
        return self.epsilon, n0, n1, self.estimate
    def rbd(self):
        "Right biased difference: returns an atomic distribution that's to the right of the real one"
        if self.estimate != "under":
            raise Exception("distribution must be underestimate")
        diff = np.copy(self.interval)
        diff[1:] -= diff[0:-1]
        #diff[0] -= 0 # assume everything to the left is zero
        return diff
    def lbd(self):
        "Left biased difference: returns an atomic distribution that's to the left of the real one"
        if self.estimate != "over":
            raise Exception("distribution must be overestimate")
        diff = -self.interval
        diff[0:-1] += self.interval[1:]
        diff[-1] += 1 # assume everything to the right is one
        return diff
    def index(self, x, reverse=False):
        "Return the index that corresponds to x (with the correct rounding to give you the right type of estimate)"
        left = (self.estimate == "under")
        if reverse: left = not left
        if left:
            j = np.floor(x/self.epsilon + self.zero)
        else:
            j = np.ceil(x/self.epsilon + self.zero)
        j = np.clip(j, 0, self.interval.shape[0]-1).astype(np.int32)
        return j
    def get(self, x):
        "Return the value that corresponds to x (with the correct rounding to give you the right type of estimate)"
        return self.interval[self.index(x)]
    def set_one_after(self, x):
        "Set the vector to 1 for entries that correspond to positions >= x (rounding depends on estimate)"
        # set reverse so that the estimate is right
        # overestimate will be floor(), underestimate will be ceil()
        i = self.index(x, reverse=True)
        self.interval[i:] = 1
    @staticmethod
    def dirac(epsilon, n0, n1, x, estimate=None):
        "Create a dirac distribution."
        dist = DistributionFunction(epsilon, n0, n1, estimate)
        dist.set_one_after(x)
        return dist
    @staticmethod
    def mix(dist, weights):
        """Mix some distributions with the given weights.
        Both dist and weights should be lists."""
        p = dist[0].params()
        for x in dist[1:]:
            assert p == x.params()
        av = sum([a.interval * b for (a, b) in zip(dist, weights)])
        return DistributionFunction(*p,
                                    copyfrom=av)
    def plot(self):
        "Use matplotlib to plot this distribution (works in Jupyter and maybe other stuff)"
        matplotlib.pyplot.figure(figsize=(16, 2))
        matplotlib.pyplot.plot(self.range(), self.interval)
        matplotlib.pyplot.show()
    def vals(self):
        """Return data for plotting.
        Return value: (list of x), (list of y)"""
        return self.range().tolist(), self.interval.tolist()
    def integrate(self, h, estimate=None, slice_=slice(0, None)):
        """Integrate a decreasing function with respect to the distribution.
        Make sure that the first coordinate of h is the minimum value over
        the whole real line, and the last coordinate is the maximum value.
        
        If self.estimate == "under", then the result is an underestimate,
        because underestimating a cumulative distribution function makes
        it look like the mass is farther right, and the decreasing function
        is going to be lower there.
        
        If self.estimate == "over", then the result is an overestimate.
        I guess? Should we set the edges to 1, 0?"""
        if self.estimate == "under":
            # h[i+1] * (interval[i+1] - interval[i])
            assert estimate == "under"
            return np.dot(h, self.rbd()[slice_])
        else:
            # h[i] * (interval[i+1] - interval[i])
            assert estimate == "over"
            return np.dot(h, self.lbd()[slice_])
    @staticmethod
    def plotboth(a, b):
        """Plot two distributions together."""
        matplotlib.pyplot.figure(figsize=(16, 2))
        for x in [a, b]: matplotlib.pyplot.plot(x.range().tolist(), x.interval.tolist())
        matplotlib.pyplot.show()

def distribution_values(epsilon, n0, n1):
    return (np.arange(n1 - n0 + 2) - 1 + n0) * epsilon

# These H-functions are defined by
# H_f(z, y) = max { x: f(exp(x), exp(y)) <= exp(z) }
# So H_+(z, y) = z + log(1 - exp(y-z)) if z > y or -inf if z <= y.

def hsum(z, y):
    # the goal is to return h[i, j] = log(exp(z[i]) - exp(y[j]))
    # or -inf if the argument of the log is negative. however, if the
    # exponent is very large, the program will be unhappy, so instead
    # we compute it as z[i] + log(1 - exp(y[j] - z[i])).
    z.shape = (z.shape[0], 1)
    H = z + np.log(1 - np.exp(y-z))
    H[np.isnan(H)] = -np.inf
    return H

def hrec(z, y):
    z.shape = (z.shape[0], 1)
    H = z - np.log(1 - np.exp(z-y))
    H[np.isnan(H)] = np.inf
    return H
    #z -log(1 - exp(z-y))
    #return a big matrix with a[i, j] = H(z[i], y[j])
    #H = np.exp(-z).reshape((z.shape[0], 1)) - np.exp(-y)
    #return -np.log(np.maximum(H, 0)) # atashi iya ne

def hp(p):
    def h(z, y):
        z.shape = (z.shape[0], 1)
        H = z + np.log(1 - np.exp(p*(y-z))) / p
        H[np.isnan(H)] = -np.inf if p > 0 else np.inf
        return H
    return h

def hmin(z, y):
    # if z > y, then inf, otherwise z
    z.shape = (z.shape[0], 1)
    # if we use z >= y here, it causes serious problems
    # with the sum-min computation, because the minimum value
    # drifts backwards. fortunately, our definitions allow
    # us to use z > y here.
    return np.where(z > y, np.inf, z)

def compute_estimates(fp, fm, H, epsilon, n0, n1, bottom, top, left, right):
    x = distribution_values(epsilon, n0, n1)
    h = H(x[bottom:top], x[left:right])
    return fp.get(h), fm.get(h) # set to zero for y = 0, -1?

# After a little while, the distributions will not be proper.
# This function turns a non-proper distribution into a proper distribution
# by setting the left coordinate to zero and the right coordinate to one.
# So the mass at infinity will reappear at the very left and the very right.

# FIXME: This isn't handled properly right now, so if there's a lot of mass
# on the two boundaries then the results will probably be wrong?

def mash(x):
    x[0] = 0
    x[-1] = 1
    return x

def recursion_for_low_memory(fp, fm, H0, H1, epsilon, n0, n1, step=1000):
    n = n1 - n0 + 2
    fp_accumulator, fm_accumulator = np.zeros(n), np.zeros(n)
    nonzero = 0
    total = 0
    for bottom in range(0, n, step):
        top = min(bottom + step, n)
        for left in range(0, n, step):
            right = min(left + step, n)
            p0 = 0.5
            p1 = 0.5
            ep_0, em_0 = compute_estimates(fp, fm, H0, epsilon, n0, n1, bottom, top, left, right)
            if np.max(ep_0) > 0:
                nonzero += 1
                fp_accumulator[bottom:top] += p0 * fp.integrate(ep_0, "over", slice_=slice(left, right))
                fm_accumulator[bottom:top] += p0 * fm.integrate(em_0, "under", slice_=slice(left, right))
            ep_0, em_0 = None, None
            ep_1, em_1 = compute_estimates(fp, fm, H1, epsilon, n0, n1, bottom, top, left, right)
            if np.max(ep_1) > 0:
                nonzero += 1
                fp_accumulator[bottom:top] += p1 * fp.integrate(ep_1, "over", slice_=slice(left, right))
                fm_accumulator[bottom:top] += p1 * fm.integrate(em_1, "under", slice_=slice(left, right))
            ep_1, em_1 = None, None
            total += 2
    print("Nonzero: %.2f" % (nonzero/total*100))
    dist = lambda x, es: DistributionFunction(epsilon, n0, n1, es, copyfrom=x)
    return dist(mash(fp_accumulator), "over"), dist(mash(fm_accumulator), "under")

def initfh(*p):
    return DistributionFunction.dirac(*p, 0, "over"), DistributionFunction.dirac(*p, 0, "under")

# epsilon, n0, n1
# e.g, if the parameters were .002, -20, 20000
# then the distribution is calculated from -0.04 to 40.0 in increments of .002
# sum-min doesn't go downward so n0 doesn't have to be large
#PARAMETERS = .002, -20, 20000
# sum-rec is symmetrical so n0 should be about -n1
#PARAMETERS = .002, -10000, 10000

PARAMETERS = 0.04, -1000, 1000
OUTPUT = "output"
ROUNDS = 400
def p_to_function(p):
    if p == -np.inf:
        return hmin
    elif p == np.inf:
        raise Exception("not implemented")
    else:
        return hp(p)
p = 1
q = -np.inf

def compute():
    #p = 0.002, -10000, 10000
    print("Parameters:\n\tepsilon = %f\n\tn0 = %d\n\tn1 = %d" % PARAMETERS)
    fp, fm = initfh(*PARAMETERS)
    print("\tp = %f\n\tq = %f" % (p, q))
    results = []
    import pickle, time
    import os.path
    outputfile = OUTPUT
    if os.path.isfile(outputfile):
        print("Output %s already exists." % outputfile)
        yn = input('Overwrite old output file? [y/N] ')
        if yn not in ["y", "Y"]:
            print("Exiting.")
            return
    else:
        print("Output file: %s" % outputfile)
    def dumpoutput():
        print("Dumping")
        t = time.time()
        fn = open(outputfile, "wb")
        pickle.dump(results, fn)
        fn.close()
        print("Dumping took %0.3f seconds" % (time.time() - t))
    for i in range(ROUNDS):
        d = fp.interval - fm.interval
        print("Iteration %d. fp - fm: max %e, min %e" % (i, np.max(d), np.min(d)))
        results.append((fp.vals(), fm.vals()))
        P = p_to_function(p)
        Q = p_to_function(q)
        fp, fm = recursion_for_low_memory(fp, fm, P, Q, PARAMETERS[0], *PARAMETERS[1:])
        dump = False
        if i>0 and i % 12 == 0:
            dump = True
            dumpoutput()
    if not dump:
        dumpoutput()

if __name__ == "__main__":
    compute()
