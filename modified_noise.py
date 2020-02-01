# Copyright 2018 Rigetti Computing
##############################################################################

from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union

import numpy as np
import sys

from pyquil.gates import I, MEASURE, X
from pyquil.quilbase import Pragma, Gate
from pyquil.quilatom import MemoryReference, format_parameter, ParameterDesignator

if TYPE_CHECKING:
    from pyquil.quil import Program
    from pyquil.api import QPUConnection, QVMConnection  # noqa: F401

INFINITY = float("inf")
"Used for infinite coherence times."
from pyquil.noise import NoiseModel,KrausModel, tensor_kraus_maps, combine_kraus_maps, damping_after_dephasing, NoisyGateUndefined, NO_NOISE, ANGLE_TOLERANCE, _get_program_gates 
from dd_sequences import get_combined_gate_representation_for_noise_model


#This function is modified to include combined gates (RX-Y, RX-Z, RZ-X, RZ-Y, CZ-XX, CZ-YY, CZ-ZZ)
def get_modified_noisy_gate(gate_name: str, params: Iterable[ParameterDesignator]) -> Tuple[np.ndarray, str]:
    """
    Look up the numerical gate representation and a proposed 'noisy' name.

    :param gate_name: The Quil gate name
    :param params: The gate parameters.
    :return: A tuple (matrix, noisy_name) with the representation of the ideal gate matrix
        and a proposed name for the noisy version.
    """
    params = tuple(params)
    if gate_name == "I":
        assert params == ()
        return np.eye(2), "NOISY-I"
    if gate_name == "RX":
        angle, = params
        if np.isclose(angle, np.pi / 2, atol=ANGLE_TOLERANCE):
            return (np.array([[1, -1j],
                              [-1j, 1]]) / np.sqrt(2),
                    "NOISY-RX-PLUS-90")
        elif np.isclose(angle, -np.pi / 2, atol=ANGLE_TOLERANCE):
            return (np.array([[1, 1j],
                              [1j, 1]]) / np.sqrt(2),
                    "NOISY-RX-MINUS-90")
        elif np.isclose(angle, np.pi, atol=ANGLE_TOLERANCE):
            return (np.array([[0, -1j],
                              [-1j, 0]]),
                    "NOISY-RX-PLUS-180")
        elif np.isclose(angle, -np.pi, atol=ANGLE_TOLERANCE):
            return (np.array([[0, 1j],
                              [1j, 0]]),
                    "NOISY-RX-MINUS-180")
    if gate_name == "RY":
        angle, = params
        if np.isclose(angle, np.pi, atol=ANGLE_TOLERANCE):
            return (np.array([[0, -1],
                              [1, 0]]),
                    "NOISY-RY-PLUS-180")
        elif np.isclose(angle, -np.pi, atol=ANGLE_TOLERANCE):
            return (np.array([[0, 1],
                              [-1, 0]]),
                    "NOISY-RY-MINUS-180")
    
    
    elif gate_name == "CZ":
        assert params == ()
        return np.diag([1, 1, 1, -1]), "NOISY-CZ"
    
    if "RX-" in gate_name:  
        matrix, new_name = get_combined_gate_representation_for_noise_model(gate_name)
        return matrix, "NOISY-" + gate_name 
    if "RZ-" in gate_name:
        matrix, new_name = get_combined_gate_representation_for_noise_model(gate_name)
        return matrix, "NOISY-" + gate_name 
    if "CZ-" in gate_name:
        matrix, new_name = get_combined_gate_representation_for_noise_model(gate_name)
        return matrix, "NOISY-" + gate_name 
    

    raise NoisyGateUndefined("Undefined gate and params: {}{}\n"
                             "Please restrict yourself to I, RX(+/-pi), RX(+/-pi/2), CZ"
                             .format(gate_name, params))


###
def _modified_decoherence_noise_model(
    gates: Sequence[Gate],
    T1: Union[Dict[int, float], float] = 30e-6,
    T2: Union[Dict[int, float], float] = 30e-6,
    gate_time_1q: float = 50e-9,
    gate_time_2q: float = 150e-09,
    ro_fidelity: Union[Dict[int, float], float] = 0.95,
) -> NoiseModel:
    """
    The default noise parameters

    - T1 = 30 us
    - T2 = 30 us
    - 1q gate time = 50 ns
    - 2q gate time = 150 ns

    are currently typical for near-term devices.

    This function will define new gates and add Kraus noise to these gates. It will translate
    the input program to use the noisy version of the gates.

    :param gates: The gates to provide the noise model for.
    :param T1: The T1 amplitude damping time either globally or in a
        dictionary indexed by qubit id. By default, this is 30 us.
    :param T2: The T2 dephasing time either globally or in a
        dictionary indexed by qubit id. By default, this is also 30 us.
    :param gate_time_1q: The duration of the one-qubit gates, namely RX(+pi/2) and RX(-pi/2).
        By default, this is 50 ns.
    :param gate_time_2q: The duration of the two-qubit gates, namely CZ.
        By default, this is 150 ns.
    :param ro_fidelity: The readout assignment fidelity
        :math:`F = (p(0|0) + p(1|1))/2` either globally or in a dictionary indexed by qubit id.
    :return: A NoiseModel with the appropriate Kraus operators defined.
    """
    all_qubits = set(sum(([t.index for t in g.qubits] for g in gates), []))
    if isinstance(T1, dict):
        all_qubits.update(T1.keys())
    if isinstance(T2, dict):
        all_qubits.update(T2.keys())
    if isinstance(ro_fidelity, dict):
        all_qubits.update(ro_fidelity.keys())

    if not isinstance(T1, dict):
        T1 = {q: T1 for q in all_qubits}

    if not isinstance(T2, dict):
        T2 = {q: T2 for q in all_qubits}

    if not isinstance(ro_fidelity, dict):
        ro_fidelity = {q: ro_fidelity for q in all_qubits}

    
    kraus_maps = []
    for g in gates:
        targets = tuple(t.index for t in g.qubits)
        key = (g.name, tuple(g.params))
        if g.name in NO_NOISE:
            if not g.dd:
                g.gate_time = gate_time_1q
            continue
        matrix, _ = get_modified_noisy_gate(g.name, g.params)

        if len(targets) == 1:
            if g.gate_time == None:
                g.gate_time = gate_time_1q
            noisy_I = damping_after_dephasing(T1.get(targets[0], INFINITY), T2.get(targets[0], INFINITY),
                                              g.gate_time)
        else:
            if len(targets) != 2:
                raise ValueError("Noisy gates on more than 2Q not currently supported")
            if g.gate_time == None:
                g.gate_time = gate_time_2q

            # note this ordering of the tensor factors is necessary due to how the QVM orders
            # the wavefunction basis
            noisy_I = tensor_kraus_maps(damping_after_dephasing(T1.get(targets[1], INFINITY),
                                                                T2.get(targets[1], INFINITY),
                                              g.gate_time),
                                        damping_after_dephasing(T1.get(targets[0], INFINITY),
                                                                T2.get(targets[0], INFINITY),
                                              g.gate_time))
        kraus_maps.append(KrausModel(g.name, tuple(g.params), targets,
                                     combine_kraus_maps(noisy_I, [matrix]),
                                     1.0))
    aprobs = {}
    for q, f_ro in ro_fidelity.items():
        aprobs[q] = np.array([[f_ro, 1. - f_ro],
                              [1. - f_ro, f_ro]])

    return NoiseModel(kraus_maps, aprobs)


def _modified_noise_model_program_header(noise_model: NoiseModel) -> "Program":
    """
    Generate the header for a pyquil Program that uses ``noise_model`` to overload noisy gates.
    The program header consists of 3 sections:

        - The ``DEFGATE`` statements that define the meaning of the newly introduced "noisy" gate
          names.
        - The ``PRAGMA ADD-KRAUS`` statements to overload these noisy gates on specific qubit
          targets with their noisy implementation.
        - THe ``PRAGMA READOUT-POVM`` statements that define the noisy readout per qubit.

    :param noise_model: The assumed noise model.
    :return: A quil Program with the noise pragmas.
    """
    from pyquil.quil import Program

    p = Program()
    defgates: Set[str] = set()
    for k in noise_model.gates:

        # obtain ideal gate matrix and new, noisy name by looking it up in the NOISY_GATES dict
        try:
            ideal_gate, new_name = get_modified_noisy_gate(k.gate, tuple(k.params))

            # if ideal version of gate has not yet been DEFGATE'd, do this
            if new_name not in defgates:
                p.defgate(new_name, ideal_gate)
                defgates.add(new_name)
        except NoisyGateUndefined:
            print(
                "WARNING: Could not find ideal gate definition for gate {}".format(k.gate),
                file=sys.stderr,
            )
            new_name = k.gate

        # define noisy version of gate on specific targets
        p.define_noisy_gate(new_name, k.targets, k.kraus_ops)

    # define noisy readouts
    for q, ap in noise_model.assignment_probs.items():
        p.define_noisy_readout(q, p00=ap[0, 0], p11=ap[1, 1])
    return p


def apply_modified_noise_model(prog: "Program", noise_model: NoiseModel) -> "Program":
    """
    Apply a noise model to a program and generated a 'noisy-fied' version of the program.

    :param prog: A Quil Program object.
    :param noise_model: A NoiseModel, either generated from an ISA or
        from a simple decoherence model.
    :return: A new program translated to a noisy gateset and with noisy readout as described by the
        noisemodel.
    """
    new_prog = _modified_noise_model_program_header(noise_model)
    for i in prog:
        if isinstance(i, Gate) and noise_model.gates:
            try:
                _, new_name = get_modified_noisy_gate(i.name, tuple(i.params))
                new_prog += Gate(new_name, [], i.qubits)
            except NoisyGateUndefined:
                new_prog += i
        else:
            new_prog += i
    return new_prog

def add_modified_decoherence_noise(
    prog: "Program",
    T1: Union[Dict[int, float], float] = 30e-6,
    T2: Union[Dict[int, float], float] = 30e-6,
    gate_time_1q: float = 50e-9,
    gate_time_2q: float = 150e-09,
    ro_fidelity: Union[Dict[int, float], float] = 0.95,
) -> "Program":
    """
    Add generic damping and dephasing noise to a program.

    This high-level function is provided as a convenience to investigate the effects of a
    generic noise model on a program. For more fine-grained control, please investigate
    the other methods available in the ``pyquil.noise`` module.

    In an attempt to closely model the QPU, noisy versions of RX(+-pi/2) and CZ are provided;
    I and parametric RZ are noiseless, and other gates are not allowed. To use this function,
    you need to compile your program to this native gate set.

    The default noise parameters

    - T1 = 30 us
    - T2 = 30 us
    - 1q gate time = 50 ns
    - 2q gate time = 150 ns

    are currently typical for near-term devices.

    This function will define new gates and add Kraus noise to these gates. It will translate
    the input program to use the noisy version of the gates.

    :param prog: A pyquil program consisting of I, RZ, CZ, and RX(+-pi/2) instructions
    :param T1: The T1 amplitude damping time either globally or in a
        dictionary indexed by qubit id. By default, this is 30 us.
    :param T2: The T2 dephasing time either globally or in a
        dictionary indexed by qubit id. By default, this is also 30 us.
    :param gate_time_1q: The duration of the one-qubit gates, namely RX(+pi/2) and RX(-pi/2).
        By default, this is 50 ns.
    :param gate_time_2q: The duration of the two-qubit gates, namely CZ.
        By default, this is 150 ns.
    :param ro_fidelity: The readout assignment fidelity
        :math:`F = (p(0|0) + p(1|1))/2` either globally or in a dictionary indexed by qubit id.
    :return: A new program with noisy operators.
    """
    gates = _get_program_gates(prog)
    #definitions = {
    #    def_gate.name: def_gate.matrix for def_gate in prog.defined_gates #figure out what is it for 
    #}
    noise_model = _modified_decoherence_noise_model(
        gates,
        T1=T1,
        T2=T2,
        gate_time_1q=gate_time_1q,
        gate_time_2q=gate_time_2q,
        ro_fidelity=ro_fidelity
    )
    return apply_modified_noise_model(prog, noise_model)
