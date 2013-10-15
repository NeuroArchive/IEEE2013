import numpy as np
import matplotlib.pyplot as plt
from traits.api import HasTraits, Int, Array, Bool
from itertools import combinations_with_replacement as comb
from itertools import product as prod
import re


def iptgen(n):
    """Generate the complete set of vector with n components

    Parameters
    ----------
    n : int
        size of all vectors

    Returns
    -------
    ipt : array int
        set of 0/1 vectors to be classified
    """
    ipt = [[int(j) for j in np.binary_repr(i, n)] for i in range(2**n)]
    return np.array(ipt, dtype=np.int)


def uniquify_function(functions, parameters):
    """Uniquify the list of function and the associated parameters
    taken from a stackoverflow thread"""
    #Uniquifying is faster if the array is contiguous
    temp = np.ascontiguousarray(functions)
    temp = temp.view(np.dtype((np.void,
                               functions.dtype.itemsize * functions.shape[1])))
    _, idx = np.unique(temp, return_index=True)

    functions = functions[idx]
    parameters = parameters[idx]

    return functions, parameters


def signal_theory_analysis(ftar, f):
    """Return the (hits, false alarm) couple given a target
    classification

    parameters
    ----------
    ftar : int array
         describe a target classification
    f : int array
        desecribe the actual function

    returns
    -------
    spe : a float between 0 and 1
    sen : a float between 0 and 1
    matt : a float between -1 and 1"""
    #Use the binarity of the vector to detect the differences
    tp = np.array(ftar)*2 - f
    TN = len(np.repeat(tp, tp == 0))
    FN = len(np.repeat(tp, tp == 2))
    TP = len(np.repeat(tp, tp == 1))
    FP = len(np.repeat(tp, tp == -1))

    spe = float(TN)/(TN+FP)
    sen = float(TP)/(TP+FN)

    return 1-spe, sen


def select_subset(n, sparsity):
    """Select a subset of input of a given sparsity

    Parameters
    ----------
    n: an int
        number of component of a vector
    sparsity: an int
        number of 1s per vector
    """
    #This trace vector enable to to pick the desired vectors
    ipt = iptgen(n)
    selection = np.sum(ipt, axis=1) == sparsity
    return selection


def select_functions(selection):
    """Create a list for selecting the column in the set of functions"""
    return [i for i, test in enumerate(selection) if test]


class PossibleSetTlu(HasTraits):
    """Wrapper of the different functions"""
    n = Int(4)
    w_min = Int(0)
    w_max = Int(3)
    t_min = Int(0)
    t_max = Int(5)
    sparsity = Int(-1)
    order = Bool(True)
    parameters = Array(dtype=np.int)
    functions = Array(dtype=np.int)
    ipt = Array(dtype=np.int)

    def __init__(self):
        """Set default parameter value"""
        self.ipt = iptgen(self.n)
        self.integration_result()
        self.threshold_integration()
        self.generating_fbp()
        self.analysis()

    def _anytrait_changed(self, name, old, new):
        """ """
        if name == "sparsity":
            self.integration_result()
            self.threshold_integration()
            self.select()
            self.generating_fbp()
            self.analysis()
            #from pdb import set_trace; set_trace()

    def integration_result(self):
        """Generate the list of all possible integration result
        and the associated parameter set
        """
        ipt = np.rot90(self.ipt)
        w_min = self.w_min
        w_max = self.w_max
        n = self.n
        order = self.order

        #Generate all possible set of vectors
        if order:
            self.weights = np.array([i for i in prod(range(w_min, w_max+1),
                                                     repeat=n)],
                                    dtype=np.int)
        else:
            self.weights = np.array([i for i in comb(range(w_min, w_max+1),
                                                     n)],
                                    dtype=np.int)
        self.integration = np.dot(self.weights, ipt)

    def threshold_integration(self):
        """Threshold the result of an integration and output the result"""
        integration = self.integration
        weights = self.weights
        t_min = self.t_min
        t_max = self.t_max
        int_0 = integration.shape[0]
        int_1 = integration.shape[1]
        #Because lists end one time before the max_threshold
        n_threshold = t_max - t_min + 1
        functions = np.zeros((int_0 * n_threshold,
                              int_1), dtype=np.int)
        par = np.zeros((int_0 * n_threshold,
                        weights.shape[1] + 1), dtype=np.int)
        for i in range(t_min, t_max + 1):
            thresholds = np.ones((weights.shape[0], 1)) * i
            i_s = i - t_min
            par[i_s*int_0:(i_s+1)*int_0, :] = np.concatenate((weights,
                                                              thresholds),
                                                             axis=1)
            functions[i_s*int_0:(i_s+1)*int_0, :] = integration >= i

        self.functions, self.parameters = uniquify_function(functions, par)

    def generating_fbp(self):
        """generate the fbp function whatever the size of
        the vector to be classified"""
        n = self.n
        s = self.sparsity
        exp0 = r"[0-1]{%d}1{%d}" % (n/2, n/2)
        exp1 = r"1{%d}[0-1]{%d}" % (n/2, n/2)
        ipt = np.arange(2**n)
        if self.sparsity >= 0:
            ipt = ipt[select_subset(n, s)]
        fbp = np.zeros(len(ipt), dtype=np.int)
        for i, cipt in enumerate(ipt):
            bin_i = np.binary_repr(cipt, n)
            if re.match(exp0, bin_i):
                fbp[i] = 1
            if re.match(exp1, bin_i):
                fbp[i] = 1
        self.fbp = fbp

    def analysis(self):
        couples = []
        for i in self.functions:
            couples.append(signal_theory_analysis(self.fbp, i))
        self.couples = np.unique(couples)

    def select_subset(self):
        """Select a subset of input of a given sparsity
        """
        selection = np.sum(self.ipt, axis=1) == self.sparsity
        return selection

    def select(self):
        """Create a list for selecting the column in the set of functions"""
        selection = self.select_subset()
        par = self.parameters
        sel = [i for i, test in enumerate(selection) if test]
        self.functions, self.parameters = uniquify_function(self.functions[:, sel],
                                                            par)

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(self.couples[:, 0], self.couples[:, 1], marker='s')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('False Alarm')
        ax.set_ylabel('Hits')
