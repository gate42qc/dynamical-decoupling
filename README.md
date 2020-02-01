# Dynamical Decoupling implemented on Pyquil

In this repository the necessary functionality for implementation of the periodic and hybrid (in this strategy the decoupling sequence is chosen randomly for each gate ) decoupling strategies and testing them.  Some pyquil functions are modified to enable noisy gates unsupported currently.

Functions and classes for implementation of dynamical decoupling are in the dd_sequences.py file. The modified functions for decoherence noise implementation with unupported gates and seperate gate timing should be imported from modified_noise.py. In the utils.py file there are given the functions which are needed for random-circuit generation, modification of pyquil programs to mimic real devices as close as possible.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
