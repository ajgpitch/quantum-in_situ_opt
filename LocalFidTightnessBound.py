"""
Script to compare the the local and global Choi fidelity measures
for a target on range of quantum systems of increasing numbers of qubits.
In all cases the local targets are a CNOT on the first two qubits and the
identity on the rest.

A random unitary is used to created a 'shifted' copy of the target for
comparison with the desired target. A parameter epsilon determines the
magnitude of the shift.

For each system a number of repetitions with different random unitaries
are completed and a average of the fidelity is calculated.
The results are printed to the console.
"""

# this version 2018 April 6
# Author: Ben Dive

import numpy as np
from qutip import Qobj, tensor, basis, identity
from functools import reduce

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

    def make_full_cNOT(self):
        components = [self.cNOT] + [self.iden] * (self.num_qubits - 2)
        product = reduce(tensor, components)
        return product.dag()

    def fidelities(self, epsilon):
        print("Calculating random unitary")
        unitary = self.random_unitary_on_target(epsilon)

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
        return np.linalg.eigh(matrix)

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

num_repeats = 5
qubits_num_range = range(3, 10)

qubit_gate_fids = []
qubit_local_fids = []
epsilon = 0.1

for num_qubits in qubits_num_range:
    print("Looking at {} qubits".format(num_qubits))
    gate_fid = []
    local_fid = []
    for k in range(num_repeats):
        fid_comparer = FidelityComparison(num_qubits)

        gate, local = fid_comparer.fidelities(epsilon)
        gate_fid.append(gate)
        local_fid.append(local)

    print(sum(gate_fid) / num_repeats)
    print(sum(local_fid) / num_repeats)
    qubit_gate_fids.append(sum(gate_fid) / num_repeats)
    qubit_local_fids.append(sum(local_fid) / num_repeats)

print("epsilon = {}".format(epsilon))
print("True gate fidelities: {}".format(qubit_gate_fids))
print("Local fidelities: {}".format(qubit_local_fids))
