import numpy as np

def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    PARAMETERS
    -----
    arrays : list of array-like
    1-D arrays to form the cartesian product of.
    out : ndarray
    Array to place the cartesian product in.

    RETURNS
    -----
    out : ndarray
    2-D array of shape (M, len(arrays)) containing cartesian products
    formed of input arrays.

    EXAMPLES
    -----
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def iptgen(n=4,ex=[2]):
    """Generate a set of input vector with n components and containing
    equal/different number of ones given ex

    Parameters
    ----------
    n : int
        number of components in each vectors
    ex : list
        numbers of 1 in each vector

    Returns
    -------
    ipt : array int
        set of 0/1 vectors to be classified

    expamples
    ---------
    >>> iptgen(n=4, ex=[2])
    array([[0,0,1,1],
           [0,1,0,1],
           [0,1,1,0],
           [1,0,0,1],
           [1 0 1 0],
           [1 1,0,0]])
    """
    #Generate the complete set of vectors using binary representation
    ipt = [[int(j) for j in np.binary_repr(i, n)] for i in range(2**n)]
    #This trace vector enable to to pick the desired vectors
    tr = np.zeros(2**n, dtype=np.int)
    for i in ex:
        tr += np.sum(ipt, axis=1) == i
    #Selecting the vectors
    ipt = np.repeat(ipt, tr >= 1, axis=0)
    return ipt


def integration_result (n, min_weight, max_weight):
    """Generate the list of all possible integration result
    and the associated parameter set

    Parameters
    ----------
    n: int
        number of input variables
    min_weight, max_weight: ints
        minimal and maximal value for the weights

    Returns
    -----
    par: a 2-d array
    syn: a 2-d array of size p x len(ipt)
    subunit: the quadruplet describing the subunit

    """
    #Generate the complete set of vectors
    ipt = iptgen(n,range(n+1))
    ipt = np.rot90(ipt)

    #Generate all possible set of vectors
    weights = cartesian(np.repeat([np.arange(min_weight, max_weight+1)],n,axis=0))
    integration = np.dot(weights, ipt)

    return integration.astype(np.int), weights.astype(np.int)


def threshold_integration(integration, weights, min_threshold, max_threshold):
    """Threshold the result of an integration and output the result"""
    int_0 = integration.shape[0]
    int_1 = integration.shape[1]
    #Because lists end one time before the max_threshold
    n_threshold = max_threshold - min_threshold + 1
    output = np.zeros((int_0 * n_threshold,
                       int_1), dtype=np.int)
    parameters = np.zeros((int_0 * n_threshold,
                           weights.shape[1] + 1), dtype=np.int)
    for i in range(min_threshold, max_threshold + 1):
        thresholds = np.ones((weights.shape[0],1)) * i
        i_s = i - min_threshold
        parameters[i_s*int_0:(i_s+1)*int_0,:] = np.concatenate((weights, thresholds), axis=1)
        output[i_s*int_0:(i_s+1)*int_0,:] = integration >= i

    return output, parameters

def uniquify_function(output, parameters):
    """uniquify the list of function and the associated parameters"""
    functions, uindex = np.unique(output.view([('',output.dtype)]*output.shape[1]),
                                    return_index=True)
    output = output[uindex]
    parameters = parameters[uindex]

    return output, parameters

def signal_theory_analysis(ftar,f):
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
    TN = len(np.repeat(tp,tp==0))
    FN = len(np.repeat(tp,tp==2))
    TP = len(np.repeat(tp,tp==1))
    FP = len(np.repeat(tp,tp==-1))

    spe = float(TN)/(TN+FP)
    sen = float(TP)/(TP+FN)

    return 1-spe, sen
