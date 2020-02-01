from typing import List

from pyquil.gates import *
from pyquil.quilbase import Gate
from pyquil.quil import Program
import random
random.seed()
import numpy as np

pi = np.pi

class gates_with_time(Gate):
    """
    Gates with an additional attribute gate time. 
    """
    dd = True
    def __init__(self, name, params, qubits):
        super().__init__(name, params, qubits)
        self.gate_time = None
        
def set_gate_time(program: "Program", gate_time: float = None) -> "Program":
    """
    :param program: A program implementing some circuit
    :param gate_time: The gate time of the gates
    :return : Program consisting of gates from gates_with_time Class
    """
    p_new = Program()
    for g in program:
        gate = gates_with_time(g.name, g.params, g.qubits)
        gate.gate_time = gate_time
        p_new +=gate
    return p_new
##########################################

def one_qubit_circuit(q_index: int, depth: int) -> "Program":
    """
    :param q_index: index of the qubit which the circuit acts
    :depth: depth of the circuit
    :return: a program corresponding to a random U 
    """
    gate_set = [RX, RZ, T]
    instructions = []
    for i in range(depth):
        g = random.choice(gate_set)
        if g is T:
            instructions.append(RZ(pi/4,q_index))
        else:
            instructions.append(g(pi/2,q_index))
    
    return Program(instructions)

def two_qubit_circuit(q_index: List[int],n_cycles:int) -> "Program":
    """
    :param q_index: indexes of the qubits which the circuit acts
    :n_cycles: number of cycles of the circuit
    :return: a program corresponding to a random U 
    """
    get_set = [RX, RZ, T]
    instructions = []
    #1. applying Hadamard's in native language
    instructions.extend([RZ(pi/2, q_index[0]),RX(pi/2, q_index[0]),RZ(pi/2, q_index[0])])
    instructions.extend([RZ(pi/2, q_index[1]),RX(pi/2, q_index[1]),RZ(pi/2, q_index[1])])
    #2. applying CZ followed by 1 qubit gates 
    for i in range(n_cycles):
        instructions.append(CZ(q_index[0],q_index[1]))
        for idx in (q_index):
            g = random.choice(get_set)
            if g is T:
                instructions.append(RZ(pi/4,idx))
            else:
                instructions.append(g(pi/2,idx))
    
    return Program(instructions)


def add_pragma_block(program: "Program") -> "Program":
    """
    :param program: A Program corresponding to some circuit.
    :return program: Program 
    """
    inst = program.instructions
    new_inst = ['PRAGMA PRESERVE_BLOCK'] + inst + ['PRAGMA END_PRESERVE_BLOCK']
    return Program(new_inst)
       
def get_wait_circuit(q_index: List[int], n: int, nI: int = 4) -> "Program":
    """
    :param q_index: index(es) of qubit(s) for inserting 'wait' blocks
    :param n:  number of wait circuits; each circuit consists of nI identity gates  
    :param nI: number of identity gates in wait block
    :return: program with wait sequence 
    """
    
    indexes = q_index
    if type(q_index) == int:
        q_index = [q_index]
    dd = []  
    for i, index in enumerate(q_index):
        dd.extend([I(index)] * (n * nI)) #it can be modified to include buffer time (I gates)
            
        
    return Program(dd)



def get_dagger_of_native(gate: Gate) -> Gate:
    """
    :param gate: A gate from native gate set
    :return:  the conjugated and transposed gate
        
    """
    if isinstance(gate, Gate):
        if gate.name == "RZ":
            return RZ(-gate.params[0], gate.qubits[0])
        if gate.name == "RX":
            return RX(-gate.params[0], gate.qubits[0])
        if gate.name == "CZ":
            return CZ(*gate.qubits)

    raise ValueError("Unsupported gate: " + str(gate))
    
def get_dagger_of_prog(program: "Program") -> "Program":
    """
    :param program: A program consisting of gates from native gate set.
    :return: The dagger program
    """
    p_new = Program()
    for gate in reversed(program.instructions):
        p_new+=(get_dagger_of_native(gate))
    return p_new


def insert_identities(program: "Program", interval_time: float =None) -> "Program":
    """
    :param program: A program consisting of gates from native gate set.
    :param interval_time: The duration of the free evolution (Identity gate)
    :return: A program with inserted identities
    
     For instance a program RZ(pi/3, 0) RX(pi/2,1) CZ (1,0) will be modified to 
                    RZ (pi/3, 0) I(0) RX(pi/2, 1) I(1), CZ(1,0) I(1), I(0)
    """
    p = set_gate_time(program, None)
    p_new = Program()
    for g in p:
        if len(g.qubits) == 2:
            interval_I2 = gates_with_time('I', [], [g.qubits[0]])
            interval_I1 = gates_with_time('I', [], [g.qubits[1]])
            interval_I1.gate_time = interval_time
            interval_I2.gate_time = interval_time
            p_new += g
            p_new += interval_I1
            p_new += interval_I2
        else: 
            interval_I = gates_with_time('I', [], g.qubits)
            interval_I.gate_time = interval_time
            p_new+= g 
            p_new+= interval_I
    return p_new
    
    
def get_zx_DD_sequence(q_index: List[int], n: int) -> "Program":
    """
    :param q_index: index(es) of qubit(s) for applying DD sequence
    :param n:  number of sequence; each sequence is consisted of ZXZX pulses
    :return: program with DD sequence 
    """
    
    indexes = q_index
    if type(q_index) == int:
        q_index = [q_index]
    dd = []  
    for i, index in enumerate(q_index):
        dd.extend([RZ(pi, index),RX(pi,index), RZ(pi,index),RX(pi,index)] * n)             
        
    return Program(dd)


def get_xy_DD_sequence(q_index: List[int], n: int) -> "Program":
    """
    :param q_index: index(es) of qubit(s) for applying DD sequence
    :param n:  number of sequence; each sequence is consisted of XYXY  pulses
    :return: program with DD sequence 
    """
    
    indexes = q_index
    if type(q_index) == int:
        q_index = [q_index]
    dd = []  
    for i, index in enumerate(q_index):
        dd.extend([RX(pi,index),RY(pi, index),RX(pi,index),RY(pi,index)] * n) 
    return Program(dd)

