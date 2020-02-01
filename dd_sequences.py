from typing import Iterable, Tuple

import numpy as np
from numpy import pi


from pyquil import Program

from pyquil.gates import RX, RY, RZ, CZ, I
from pyquil.noise import  ANGLE_TOLERANCE, NoisyGateUndefined
from pyquil.quilbase import DefGate, Gate
from pyquil.quilatom import ParameterDesignator

from utils import gates_with_time, set_gate_time

BASE_GATE_MATRICES = {
    'RX': lambda phi: np.array([[np.cos(phi / 2), -1j * np.sin(phi / 2)],
                                [-1j * np.sin(phi / 2), np.cos(phi / 2)]]),
    'RZ': lambda phi: np.array([[np.cos(phi / 2) - 1j * np.sin(phi / 2), 0],
                                [0, np.cos(phi / 2) + 1j * np.sin(phi / 2)]]),
    'RY': lambda phi: np.array([[np.cos(phi / 2), -np.sin(phi / 2)],
                                [np.sin(phi / 2), np.cos(phi / 2)]]),
    'CZ': lambda: np.array([[[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, -1]]]),
}

Z_MATRIX = np.array([
    [1, 0],
    [0, -1]
])

Z_MATRIX2 = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

Y_MATRIX = np.array([
    [0, -1],
    [1, 0]    
])

Y_MATRIX2 = np.array([
    [ 0,  0,  0,  1],
    [ 0,  0, -1,  0],
    [ 0, -1,  0,  0],
    [ 1,  0,  0,  0]
])

X_MATRIX = np.array([
    [0, -1j],
    [-1j, 0]    
])

X_MATRIX2 = np.array([
    [ 0,  0, 0, -1],
    [ 0,  0, -1,  0],
    [ 0, -1,  0,  0],
    [ -1,  0,  0,  0]
])

matrices = {"X": X_MATRIX, "XX": X_MATRIX2, "Y": Y_MATRIX, "YY": Y_MATRIX2, "Z": Z_MATRIX, "ZZ": Z_MATRIX2}

##########################################


def check_rx_angle(gate_name: str, params: Iterable[ParameterDesignator], ANGLE_TOLERANCE) -> Tuple[np.ndarray, str]:
    """
    :param str gate_name: The Quil gate name
    :param str pulse: The pulse name to combine with the gate 
    :param [float] params: The gate parameters.
    :return: A tuple (matrix, noisy_name) with the representation of the ideal gate matrix
            and a proposed name for the noisy version.
        :rtype: Tuple[np.array, str]
        """
    if gate_name == 'RX':
        angle, = params
        angle += pi 
        if np.isclose(angle, 3*np.pi / 2, atol=ANGLE_TOLERANCE):
            return BASE_GATE_MATRICES[gate_name](angle), gate_name + "-PLUS-270"
        
        elif np.isclose(angle, 2*np.pi , atol=ANGLE_TOLERANCE):
            return  np.eye(2), gate_name + "-PLUS-360"

        elif np.isclose(angle, 0, atol=ANGLE_TOLERANCE):
            return  np.eye(2), gate_name + "-0"


        elif np.isclose(angle, np.pi / 2, atol=ANGLE_TOLERANCE):
            return np.array([[1, -1j], 
                             [-1j, 1]]) / np.sqrt(2), gate_name + "-PLUS-90"

        elif np.isclose(angle, np.pi, atol=ANGLE_TOLERANCE):
            return np.array([[0, -1j],
                            [-1j, 0]]), gate_name + "-PLUS-180"     
            
    raise NoisyGateUndefined("Undefined gate and params: {} {}\n"
                                 "Please restrict yourself to I, RX(+/-pi), RX(+/-pi/2)"
                                 .format(gate_name, params))

def get_combined_gate_representation(gate_name: str, pulse: str, params = None) -> Tuple[np.ndarray, str]:
        """
    
        :param str gate_name: The Quil gate name
        :param str pulse: The pulse name to combine with the gate 
        :param [float] params: The gate parameters.
        :return: A tuple (matrix, noisy_name) with the representation of the ideal gate matrix
            and a proposed name for the noisy version.
        :rtype: Tuple[np.array, str]
        """
        assert pulse == 'X' or pulse == 'Y' or pulse == 'Z', "Please provide pulses from ('X', 'Y', 'Z')"
        if gate_name == "CZ":
            return np.matmul(np.diag([1, 1, 1, -1]), matrices[2*pulse]), "CZ-" + pulse*2
        else:
            matrix = None
            new_name = None
            if gate_name == 'RX':
                if pulse == 'X':
                    return check_rx_angle(gate_name, params, ANGLE_TOLERANCE)
                else:
                    angle, = params
                    if np.isclose(angle, np.pi / 2, atol=ANGLE_TOLERANCE):
                        matrix, new_name = np.array([[1, -1j], 
                                         [-1j, 1]]) / np.sqrt(2), gate_name + '-' +pulse+ "-PLUS-90"

                    elif np.isclose(angle, -np.pi / 2, atol=ANGLE_TOLERANCE):
                        matrix, new_name = np.array([[1, 1j], 
                                        [1j, 1]]) / np.sqrt(2), gate_name + '-' +pulse+ "-MINUS-90"
                    elif np.isclose(angle, np.pi, atol=ANGLE_TOLERANCE):
                        matrix, new_name = np.array([[0, -1j],
                                        [-1j, 0]]), gate_name + '-' +pulse+ "-PLUS-180"
                    elif np.isclose(angle, -np.pi, atol=ANGLE_TOLERANCE):
                        matrix, new_name = np.array([[0, 1j], 
                                        [1j, 0]]), gate_name + '-' +pulse+ "-MINUS-180"
                
            elif gate_name == 'I':
                matrix, new_name = np.eye(2), "I-" + pulse
                
            elif gate_name == 'RZ':
                angle, = params
                matrix, new_name = (BASE_GATE_MATRICES['RZ'](angle), 
                                    'RZ-' + pulse + "_" + str(int(angle/pi * 180))) 
        
        if matrix is not None:
            return np.matmul(matrix, matrices[pulse]), new_name

        raise NoisyGateUndefined("Undefined gate and params: {} {}\n"
                                 "Please restrict yourself to I, RX(+/-pi), RX(+/-pi/2), CZ"
                                 .format(gate_name, params))


# The following functions are needed for the get_modified_noisy_gate function in modified.noise module.
def check_rx_combined_gates(gate_name: str) -> Tuple[np.ndarray, str]:
    """
    :param str gate_name: The name of RX+Pulse combined gate  
    :return: A tuple (matrix, noisy_name) with the representation of the combined gate matrix
            and name
    :rtype: Tuple[np.ndarray, str]
    """
    if gate_name == "RX-Z-PLUS-90":
        return get_combined_gate_representation('RX', pulse = 'Z', params = [pi/2])
    elif gate_name == "RX-Z-MINUS-90":
        return get_combined_gate_representation('RX', pulse = 'Z', params = [-pi/2])

    elif gate_name == "RX-Z-PLUS-180":
        return get_combined_gate_representation('RX', pulse = 'Z', params = [pi])

    elif gate_name == "RX-Z-MINUS-180":
        return get_combined_gate_representation('RX', pulse = 'Z', params = [-pi])
    elif gate_name == "RX-Y-PLUS-90":
        return get_combined_gate_representation('RX', pulse = 'Y', params = [pi/2])
    elif gate_name == "RX-Y-MINUS-90":
        return get_combined_gate_representation('RX', pulse = 'Y', params = [-pi/2])

    elif gate_name == "RX-Y-PLUS-180":
        return get_combined_gate_representation('RX', pulse = 'Y', params = [pi])

    elif gate_name == "RX-Y-MINUS-180":
        return get_combined_gate_representation('RX', pulse = 'Y', params = [-pi])
    elif gate_name == "RX-0":
        return get_combined_gate_representation('RX', pulse = 'X', params = [-pi])
    elif gate_name == "RX-PLUS-90":
        return get_combined_gate_representation('RX', pulse = 'X', params = [-pi/2])
    elif gate_name == "RX-PLUS-180":
        return get_combined_gate_representation('RX', pulse = 'X', params = [0])
    elif gate_name == "RX-PLUS-270":
        return get_combined_gate_representation('RX', pulse = 'X', params = [pi/2])
    elif gate_name == "RX-PLUS-360":
        return get_combined_gate_representation('RX', pulse = 'X', params = [pi])
    
    
    raise NoisyGateUndefined("Undefined gate and params: {} \n"
                             "Please restrict yourself to I, RX(+/-pi), RX(+/-pi/2), CZ"
                             .format(gate_name))
                             
def get_combined_gate_representation_for_noise_model(gate_name: str) -> Tuple[np.ndarray, str]:
    
    """
    :param name: The name of the combined gate
    :return: Combined gate representation as [matrix, name]
            
    :rtype: Tuple
    """

    if gate_name[:3] == 'RX-':
        return check_rx_combined_gates(gate_name)
    elif gate_name[:3] == 'RZ-':
        return get_combined_gate_representation(gate_name.split('-')[0], gate_name.split('-')[1][0],
                                                [(float(gate_name.split('_')[-1])/180) *pi])
    elif gate_name[:3] == 'CZ-':
        return get_combined_gate_representation(gate_name.split('-')[0], gate_name.split('-')[1][0])
    
    raise NoisyGateUndefined("Undefined gate and params: {}\n"
                             "Please restrict yourself to I, RX(+/-pi), RX(+/-pi/2), CZ"
                             .format(gate_name))



def get_combined_gate(matrix: np.ndarray, name: str) -> Gate:
    
    """
    :param matrix: The matrix of the combined gate  
    :param name: The of the combined gate
    :return: A Gate (matrix, noisy_name) corresponding to the representation [matrix, name]
            
    :rtype: Gate
    """

    p = Program()
    combined_gate_definition = DefGate(name, matrix)
    p.inst(combined_gate_definition)
    combined_gate = combined_gate_definition.get_constructor()
    return combined_gate    

    
    
# The following classes have the same structure, they help to generate DD sequences for a given gate.


class ZXZX:
    def get_decoupling_sequence(self, gate: Gate, dd_pulse_time: float = None)  -> "Program":
        if isinstance(gate, Gate):
            angle = None

            if gate.name == "RZ":
                combined_gate = RZ
                angle, = gate.params 
                angle += pi
            else:
                combined_gate = get_combined_gate(*get_combined_gate_representation(gate.name, 'Z', gate.params))

            if len(gate.qubits)!= 1:
                p = Program(RX(pi, gate.qubits[0]),
                        RX(pi, gate.qubits[1]),
                        RZ(pi, gate.qubits[0]),
                        RZ(pi, gate.qubits[1]),
                        RX(pi, gate.qubits[0]),
                        RX(pi, gate.qubits[1]))                    

            else:
                p = Program(RX(pi, *gate.qubits),
                            RZ(pi, *gate.qubits),
                            RX(pi, *gate.qubits))

            seq = set_gate_time(p, dd_pulse_time) 
            GZ = combined_gate(angle, *gate.qubits) if angle is not None else combined_gate(*gate.qubits)
            GZ = gates_with_time(GZ.name, GZ.params, GZ.qubits)
            GZ.dd = False
            seq += GZ

            return seq
        
#####################################

class ZYZY:

    def get_decoupling_sequence(self, gate: Gate, dd_pulse_time: float = None)  -> "Program":
        if isinstance(gate, Gate):
            angle = None

            if gate.name == "RZ":
                combined_gate = RZ
                angle, = gate.params 
                angle += pi
            else:
                combined_gate = get_combined_gate(*get_combined_gate_representation(gate.name, 'Z', gate.params))

            if len(gate.qubits)!= 1:
                p = Program(RY(pi, gate.qubits[0]),
                        RY(pi, gate.qubits[1]),
                        RZ(pi, gate.qubits[0]),
                        RZ(pi, gate.qubits[1]),
                        RY(pi, gate.qubits[0]),
                        RY(pi, gate.qubits[1]))                    

            else:
                p = Program(RY(pi, *gate.qubits),
                            RZ(pi, *gate.qubits),
                            RY(pi, *gate.qubits))

            seq = set_gate_time(p, dd_pulse_time) 
            GZ = combined_gate(angle, *gate.qubits) if angle is not None else combined_gate(*gate.qubits)
            GZ = gates_with_time(GZ.name, GZ.params, GZ.qubits)
            GZ.dd = False
            seq += GZ

            return seq
#####################################
class XZXZ:
    
    def get_decoupling_sequence(self, gate: Gate, dd_pulse_time: float = None)  -> "Program":
        if isinstance(gate, Gate):
            combined_gate = get_combined_gate(*get_combined_gate_representation(gate.name, 'X', gate.params))
            if len(gate.qubits)!= 1:
                p = Program(RZ(pi, gate.qubits[0]),
                        RZ(pi, gate.qubits[1]),
                        RX(pi, gate.qubits[0]),
                        RX(pi, gate.qubits[1]),
                        RZ(pi, gate.qubits[0]),
                        RZ(pi, gate.qubits[1]))

            else:
                p = Program(RZ(pi, *gate.qubits),
                            RX(pi, *gate.qubits),
                            RZ(pi, *gate.qubits))
                           
            seq = set_gate_time(p, dd_pulse_time) 
            GX = combined_gate(*gate.qubits)
            GX = gates_with_time(GX.name, GX.params, GX.qubits)
            GX.dd = False
            seq += GX
                    
        return seq
##################################################
class XYXY:    
    def get_decoupling_sequence(self, gate: Gate, dd_pulse_time: float = None)  -> "Program":
        if isinstance(gate, Gate):
            combined_gate = get_combined_gate(*get_combined_gate_representation(gate.name, 'X', gate.params))
            if len(gate.qubits)!= 1:
                p = Program(RY(pi, gate.qubits[0]),
                        RY(pi, gate.qubits[1]),
                        RX(pi, gate.qubits[0]),
                        RX(pi, gate.qubits[1]),
                        RY(pi, gate.qubits[0]),
                        RY(pi, gate.qubits[1]))

            else:
                p = Program(RY(pi, *gate.qubits),
                            RX(pi, *gate.qubits),
                            RY(pi, *gate.qubits))
                           
            seq = set_gate_time(p, dd_pulse_time) 
            GX = combined_gate(*gate.qubits)
            GX = gates_with_time(GX.name, GX.params, GX.qubits)
            GX.dd = False
            seq += GX
                    
            return seq

##################################################
class YZYZ:
    def get_decoupling_sequence(self, gate: Gate, dd_pulse_time: float = None)  -> "Program":
        if isinstance(gate, Gate):
            angle = None
            combined_gate = get_combined_gate(*get_combined_gate_representation(gate.name, 'Y', gate.params))

            if len(gate.qubits)!= 1:
                p = Program(RZ(pi, gate.qubits[0]),
                        RZ(pi, gate.qubits[1]),
                        RY(pi, gate.qubits[0]),
                        RY(pi, gate.qubits[1]),
                        RZ(pi, gate.qubits[0]),
                        RZ(pi, gate.qubits[1]))                    

            else:
                p = Program(RZ(pi, *gate.qubits),
                            RY(pi, *gate.qubits),
                            RZ(pi, *gate.qubits))

            seq = set_gate_time(p, dd_pulse_time) 
            GY = combined_gate(angle, *gate.qubits) if angle is not None else combined_gate(*gate.qubits)
            GY = gates_with_time(GY.name, GY.params, GY.qubits)
            GY.dd = False
            seq += GY

            return seq
##########################################################

class YXYX:
    def get_decoupling_sequence(self, gate: Gate, dd_pulse_time: float = None)  -> "Program":
        if isinstance(gate, Gate):
            angle = None
            combined_gate = get_combined_gate(*get_combined_gate_representation(gate.name, 'Y', gate.params))

            if len(gate.qubits)!= 1:
                p = Program(RX(pi, gate.qubits[0]),
                        RX(pi, gate.qubits[1]),
                        RY(pi, gate.qubits[0]),
                        RY(pi, gate.qubits[1]),
                        RX(pi, gate.qubits[0]),
                        RX(pi, gate.qubits[1]))                    

            else:
                p = Program(RX(pi, *gate.qubits),
                            RY(pi, *gate.qubits),
                            RX(pi, *gate.qubits))

            seq = set_gate_time(p, dd_pulse_time) 
            GY = combined_gate(angle, *gate.qubits) if angle is not None else combined_gate(*gate.qubits)
            GY = gates_with_time(GY.name, GY.params, GY.qubits)
            GY.dd = False
            seq += GY

            return seq


####################

def get_dd_protected_program(program: "Program", sequence: "ddsequence",  gate_time: float =None, dd_pulse_time: float =None) -> "Program":
    
    """
    :param program: A program with native gates  
    :param gate_time : The duration of the gates implementing computation.  
    :return : A DD sequence inserted program: for instance, for the ZXZX 
              sequence each gate in the program is replaced by G*Z-X-Z-X sequence. 
    """
    
    p_new = Program()
    
    for g in program:
        seq = sequence().get_decoupling_sequence(g, dd_pulse_time)
        p_new += seq
    return p_new


def insert_identities_into_dd_protected_program(program: "Program",  interval_time: float =None) -> "Program":
    """
    
    :param program: A program with DD-protected Gates
    :param interval time : The duration of the free evolution (gate time of the I gates). 
    :return : A Program with inserted Identity gates
    
    For instance- each G.Z-X-Z-X sequence becomes G.Z-I-X-I-Z-I-X-I
    """
    p_new = Program()
    for g in program:
        if len(g.qubits) == 2:
            interval_I1 = gates_with_time('I', [], [g.qubits[0]])
            interval_I2 = gates_with_time('I', [], [g.qubits[1]])
            interval_I1.gate_time = interval_time
            interval_I2.gate_time = interval_time
            p_new+= interval_I1
            p_new+= interval_I2
            p_new += g
        else:
            interval_I = gates_with_time('I', [], g.qubits)
            interval_I.gate_time = interval_time
            p_new+= interval_I
            p_new+= g
    return p_new