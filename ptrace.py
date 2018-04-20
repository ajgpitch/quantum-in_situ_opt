"""
This is a modified version of the QuTiP partial trace.
It supports dense matrix as input and output.
It also separates the calculation of the permutation matrix,
which is the costly part. This allows the re-use of the permutation
matrix, which in the QSO optimisation is very frequent by the local
Choi fidelity computer.
"""

# this version 2018 April 6
# Author: Alexander Pitchford

# Since this was developed a Cythonised version of ptrace has replaced
# the version this was based upon. It is much faster, but does not support
# dense matrices or perm re-use.

__all__ = []

import numpy as np
import scipy.sparse as sp
from qutip.sparse import sp_reshape

def partial_trace(rho_data, rho_dims, sel, perm=None):
    """Compute the partial trace of the matrix

    Parameters
    ----------
    rho_data : csr_matrix or ndarray
        Matrix for which the partial trace will be computed

    rho_dims : list
        Subsystem dimensions

    sel : int/list
        An ``int`` or ``list`` of components to keep after partial trace.

    perm : ndarray
        Optionally a pre computed permutation matrix can be provided
        This may be more efficient if the same partial trace (sel)
        is to be computed for matices of the same sub-system dimensions.
        It can be calculated using the calc_perm function

    Returns
    -------
    rho1_data : qobj
        Matrix representing partial trace with selected components remaining.
    rho1_dims : list
        Subsystem dimensions for output matrix
    rho1_shape : tuple
        Shape of output matrix
    """

    #print("rho_dims {} (type: {})".format(rho_dims, type(rho_dims)))
    #print("sel {} (type: {})".format(sel, type(sel)))
    if isinstance(sel, int):
        sel = np.array([sel])
    else:
        sel = np.asarray(sel)

    if (sel < 0).any() or (sel >= len(rho_dims[0])).any():
        raise TypeError("Invalid selection index in ptrace.")

    sparse = sp.issparse(rho_data)

    if np.prod(rho_dims[1]) == 1:
        rho_data = rho_data.dot(rho_data.T.conj())

    rho1_dims = [np.asarray(rho_dims[0]).take(sel).tolist(),
                np.asarray(rho_dims[0]).take(sel).tolist()]
    rho1_shape = (np.prod(rho1_dims[0]), np.prod(rho1_dims[1]))

    if perm is None:
        perm = calc_perm(rho_dims, sel, sparse=sparse)

    if sparse:
        rhdata = perm * sp_reshape(rho_data, [np.prod(rho_dims[0])**2, 1])
        rhdata = rhdata.tolil().reshape(rho1_shape)
        rho1_data = rhdata.tocsr()
    else:
        rho1_data = np.reshape(np.sum(np.reshape(
            rho_data.flatten()[perm.flatten()], perm.shape), 1), rho1_shape)

    return rho1_data, rho1_dims, rho1_shape

def calc_perm(rho_dims, sel, sparse=False):
    """
    Calculate the permutation matrix.
    These are the indexes of the elemnts to be summed in the partial trace
    Each row holds the indexes to be summed for one element of the
    selected operator(s)
    """
    if isinstance(sel, int):
        sel = np.array([sel])
    else:
        sel = np.asarray(sel)

    if (sel < 0).any() or (sel >= len(rho_dims[0])).any():
        raise TypeError("Invalid selection index in ptrace.")

    drho = rho_dims[0]
    N = np.prod(drho)
    M = np.prod(np.asarray(drho).take(sel))
    # all elements in range(len(drho)) not in sel set
    rest = np.setdiff1d(np.arange(len(drho)), sel)
    ilistsel = _select(sel, drho)
    indsel = _list2ind(ilistsel, drho)
    ilistrest = _select(rest, drho)
    indrest = _list2ind(ilistrest, drho)
    irest = (indrest - 1) * N + indrest - 2

    perma = np.array(
                [(irest + (indsel[int(np.floor(m / M))] - 1) * N +
                 indsel[int(np.mod(m, M))]).T[0]
                 for m in range(M**2)])

    if not sparse:
        return perma

    perm = sp.lil_matrix((M**2, N**2))
    perm.rows = perma
    perm.data = np.ones_like(perm.rows)

    return perm.tocsr()

def _list2ind(ilist, dims):
    """!
    Private function returning indicies
    """
    ilist = np.asarray(ilist)
    dims = np.asarray(dims)
    irev = np.fliplr(ilist) - 1
    fact = np.append(np.array([1]), (np.cumprod(np.flipud(dims)[:-1])))
    fact = fact.reshape(len(fact), 1)
    return np.array(np.sort(np.dot(irev, fact) + 1, 0), dtype=int)


def _select(sel, dims):
    """
    Private function finding selected components
    """
    sel = np.asarray(sel)  # make sure sel is np.array
    dims = np.asarray(dims)  # make sure dims is np.array
    rlst = dims.take(sel)
    rprod = np.prod(rlst)
    ilist = np.ones((rprod, len(dims)), dtype=int)
    counter = np.arange(rprod)
    for k in range(len(sel)):
        ilist[:, sel[k]] = np.remainder(
            np.fix(counter / np.prod(dims[sel[k + 1:]])), dims[sel[k]]) + 1
    return ilist
