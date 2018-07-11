Edit the parameters to get it to do something different.

To make it more accurate:
PARAMETERS = 0.004, -10000, 10000

To change the values of p and q:
(P_, Q) = 1, np.inf

The output file is a pickled array with one entry per round.
Entries are of the form
    ((np.array(xes for upper bound), np.array(yes for upper bound)),
     (np.array(xes for lower bound), np.array(yes for lower bound)))

How to plot the upper and lower bounds

import pickle
import matplotlib.pyplot as p
q = pickle.load(open("output", "rb"))
p.plot(*q[399][0]) # 399th round
p.plot(*q[399][1])
p.show()
