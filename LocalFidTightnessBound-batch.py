"""
Script to compare the the local and global Choi fidelity measures
for a target on a quantum system with a given number of qubits.
The local targets are a CNOT on the first two qubits and the
identity on the rest.

A random unitary is used to created a 'shifted' copy of the target for
comparison with the desired target. A parameter epsilon determines the
magnitude of the shift.

A number of repetitions with different random unitaries can be specified,
which are completed and a average of the fidelity is calculated.
The results are printed to the console.

This script is designed to be used on a cluster so that large systems
can be processed. Hence cmd line args can be used to specify the number
of qubits, reps and threads
"""

# this version 2018 April 6
# Authors: Ben Dive & Alexander Pitchford

import numpy as np
import scipy.linalg as la
from qutip import Qobj, tensor, identity
from functools import reduce
import argparse
import signal

def sigterm_handler(_signo, _stack_frame):
    print("Terminated")
    raise KeyboardInterrupt("Terminated")
signal.signal(signal.SIGTERM, sigterm_handler)

NUM_THREADS = 1
NUM_QUBITS = 3
NUM_REPEATS = 1

# read in some command line args
parser = argparse.ArgumentParser(description="Command line argument parser")
parser.add_argument('-m', '--num_threads', type=int, default=0,
                                help="Number of threads")
parser.add_argument('-q', '--num_qubits', type=int, default=0,
                                help="Number of qubits")
parser.add_argument('-r', '--num_repeats', type=int, default=0,
                                help="Number of repeats")

args = vars(parser.parse_args())

if args['num_threads'] > 0:
    num_threads = args['num_threads']
else:
    num_threads = NUM_THREADS
print("{} threads specified".format(num_threads))

if args['num_qubits'] > 0:
    num_qubits = args['num_qubits']
else:
    num_qubits = NUM_QUBITS
print("{} qubits specified".format(num_qubits))

if args['num_repeats'] > 0:
    num_repeats = args['num_repeats']
else:
    num_repeats = NUM_REPEATS
print("{} repeats specified".format(num_repeats))

try:
    import mkl
    use_mkl = True
except:
    use_mkl = False

if use_mkl:
    mkl.set_num_threads(num_threads)
    print("Number of threads is {}".format(mkl.get_max_threads()))
else:
    print("mkl unavailable")


class FidelityComparison:
    cNOT = Qobj([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])
    cNOT.dims = [[2, 2], [2, 2]]
    iden = identity(2)

    def __init__(self, num_qubits):
        print("Starting init")
        self.num_qubits = num_qubits
        self.dim = 2 ** self.num_qubits

        self.full_cNOT = self.make_full_cNOT()
        print("full cNOT dims: {}".format(self.full_cNOT.dims))

    def make_full_cNOT(self):
        components = [self.cNOT] + [self.iden] * (self.num_qubits - 2)
        product = reduce(tensor, components)
        return product.dag()

    def fidelities(self, epsilon):
        print("Calculating random unitary")
        unitary = self.random_unitary_on_target(epsilon)
        print("unitary dims: {}".format(unitary.dims))

        print("Calculating gate fidelity")
        gate_fid = self.gate_fidelity(unitary)

        print("Calculating local fidelity")
        local_fid = self.local_fidelity(unitary)
        return gate_fid, local_fid

    def random_unitary_on_target(self, epsilon):
        rand_u = self.random_unitary(epsilon)
        qobj = Qobj(rand_u)
        qobj.dims = [[2] * self.num_qubits, [2] * self.num_qubits]
        return qobj * self.full_cNOT

    def random_unitary(self, epsilon):
        vals, vecs = self.make_unnormalised_random_ham()
        norm = max([abs(val) for val in vals])
        exp_vals = np.diag(
            [np.exp(1j * val * epsilon / norm) for val in vals])
        return np.dot(vecs, np.dot(exp_vals, vecs.conj().T))

    def make_unnormalised_random_ham(self):
        matrix = (np.random.normal(size=(self.dim, self.dim)) +
                  1j * np.random.normal(size=(self.dim, self.dim)))
        matrix = matrix + matrix.conj().T
        return la.eigh(matrix)

    def gate_fidelity(self, unitary):
        product = unitary.dag() * self.full_cNOT
        trace = product.tr() / self.dim
        return abs(trace) ** 2

    def local_fidelity(self, unitary):
        fidelity = 1

        print("\tOn subsystem 0,1")
        fidelity -= 1 - self.sub_fidelity_01(unitary)

        for k in range(2, self.num_qubits):
            print("\tOn subsystem {}".format(k))
            fidelity -= 1 - self.sub_fidelity_k(unitary, k)

        return fidelity

    def sub_fidelity_01(self, unitary):
        pseudo_fid = self.pseudo_fidelity_01(unitary)
        inside_trace = pseudo_fid.dag() * pseudo_fid
        return inside_trace.tr() / (4 * self.dim)

    def pseudo_fidelity_01(self, unitary):
        overlap = self.full_cNOT * unitary
        qubits_remaining = range(2, self.num_qubits)
        return overlap.ptrace(qubits_remaining)

    def sub_fidelity_k(self, unitary, k):
        pseudo_fid = self.pseudo_fidelity_k(unitary, k)
        inside_trace = pseudo_fid.dag() * pseudo_fid
        return inside_trace.tr() / (2 * self.dim)

    def pseudo_fidelity_k(self, unitary, k):
        # overlap = self.target_id * unitary
        qubits_remaining = list(range(self.num_qubits))
        qubits_remaining.remove(k)
        return unitary.ptrace(qubits_remaining)

    # Alternative to above, does not require saving rand_ham

epsilon = 0.1

print("Looking at {} qubits".format(num_qubits))
gate_fid = []
local_fid = []
for k in range(num_repeats):
    fid_comparer = FidelityComparison(num_qubits)
    gate, local = fid_comparer.fidelities(epsilon)
    gate_fid.append(gate)
    local_fid.append(local)

print("\n***********Results*************\n")
print("num qubits = {}".format(num_qubits))
print("num repeats = {}".format(num_repeats))
print("epsilon = {}".format(epsilon))
print("True gate fidelity: {}".format(sum(gate_fid) / num_repeats))
print("Local fidelity: {}".format(sum(local_fid) / num_repeats))

