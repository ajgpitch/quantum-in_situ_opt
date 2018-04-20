Simulation of *in situ* optimisation of gates in a quantum simulator
====================================================================
[Alexander Pitchford](http://github.com/ajgpitch), [Benjamin Dive](mailto:benjamindive@gmail.com)

The code in this repository can be used to generate and analyse the data that produce the numerical results in the paper 
[In situ upgrade of quantum simulators to universal computers](https://arxiv.org/abs/1701.01723). 
In the paper we present some numerical results that support the viability and scalability of a method for optimising quantum gates within the quantum simulator itself using a local Choi fidelity measure in the optimisation scheme.

Requirements
------------
This code has primarily been developed and tested with Python 3.6 on a Linux operating system. There is no reason why it should not work on other OS or Python versions. However, the multiprocessing features will not work so well on MS Windows.

[QuTiP](qutip.org) and its prerequistes are required to run most of the scripts and notebooks. It has been tested with version 4.2 has been. Older versions may work as well.

[Jupyter notebook](jupyter.org) is required to run the notebooks.

Downloading
-----------
You should clone or download this repository to use it.

Installation
------------
There are no installation steps. Simply save the files on a computer. There are no extensions that need compiling.

License
-------
You are free to use this software in your research or other activities, with or without modification, provided that the conditions listed in the LICENSE file are satisfied.
We politely request that you acknowledge its source and authors, and cite our paper in any publications that may arise from its use.

Contents and usage
------------------
Most of the code is organised into modules. There are then Python scripts and notebooks that can be run.

[qso-n_qubit-CNOT script](qso-n_qubit-CNOT.py)
This script will perform the optimisation of a CNOT on quantum system that can be configured in a wide variety of topologies and interaction types. Details of its actions and options are given in the main docstring of the file.

It requires a parameter file. By default it will use [params-quant_self_opt.ini](params-quant_self_opt.ini). So to run it with the default parameters then, in a console, enter:

```
$ python qso-n_qubit-CNOT.py 
```

The parameters all link to object attributes. There are described where they are first set to their default values in [qsoconfig.py](qsoconfig.py).

There are other parameter files provided. [qsoconfig.py](qsoconfig.py) is simply a copy of the default file. [params-4qubit-ring-heisen.ini](params-4qubit-ring-heisen.ini) is an example with different number of qubits, topology and iteraction type. To run with these parameters:

```
$ python qso-n_qubit-CNOT.py -p params-4qubit-ring-heisen.ini
```

[params-3qubit-chain-ising_equal-xy_ctrl-cNOT1-sens-nq3fet1e-2.ini](params-3qubit-chain-ising_equal-xy_ctrl-cNOT1-sens-nq3fet1e-2.ini) will run an automated search for the numerical accuracy threshold. This invloves many more pulse optimisations than they othe options and hence may take a long time to run on some systems. Again the `-p` option can be employed to use these parameters.

Copies can be made of the parameter files and they can be selected using the `-p` option.



