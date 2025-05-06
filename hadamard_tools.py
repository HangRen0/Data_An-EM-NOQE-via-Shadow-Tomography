import numpy as np
from mitiq import zne
from qiskit import QuantumCircuit, execute, Aer
from qiskit.compiler import transpile
import qiskit.providers.aer.noise as noise
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
from qiskit.providers.aer.noise import NoiseModel
from scipy.linalg import eigh
from mitiq.zne.inference import LinearFactory, RichardsonFactory, PolyFactory
from datetime import datetime

# simplified pauli measurement label and coefficient
simplified_h_diag_measure_list = [1, 11]
simplified_h_offdiag_measure_list = [0, 1, 3, 5, 11]
h_diag_coefficient_list = [-0.1779291653897952, 0.36978265527106347, -0.6365898571875489]
h_off_coefficient_list = [-0.6365898561329018, -0.17792916538979606, 0.153506530287386, 0.1535065227735808, 0.36978266455828424]

qc_s_offdiagonal_qasm =  QuantumCircuit().from_qasm_str(open('qc_transpiled.qasm', 'r').read())
qc_s_offdiagonal = transpile(qc_s_offdiagonal_qasm, optimization_level=3, basis_gates=['u3','cx'])

# circuit for h offdiagonal
qc_h_offdiag_qasm = []
for i in simplified_h_offdiag_measure_list:
    qc_h_offdiag_qasm.append(QuantumCircuit().from_qasm_str(open('hamiltonian_term/hamiltonianterm'+str(i)+'.qasm', 'r').read()))

qc_h_offdiag_list = []
for qc in qc_h_offdiag_qasm:
    qc_h_offdiag_list.append(transpile(qc, optimization_level=3, basis_gates=['cx','u3']))

#circuit for h diagonal, using 4 qubits, simplified version
qc_h_diag_qasm = []
for i in simplified_h_diag_measure_list:
    qc_h_diag_qasm.append(QuantumCircuit().from_qasm_str(open('hamiltonian_term/hamiltonianterm_diagonal'+str(i)+'.qasm', 'r').read()))

qc_h_diag_list = []
for qc in qc_h_diag_qasm:
    qc_h_diag_list.append(transpile(qc, optimization_level=3, basis_gates=['cx','u3']))

backend = Aer.get_backend('qasm_simulator')

def tell_signal(measure_str):
    if measure_str.count('0') % 2 == 0:
        return 'minus'
    else:
        return 'plus'

# tell the expectation value
def tell_exp(result):
    tem_dic_keys = list(result.get_counts().keys())
    tem_dic_values = list(result.get_counts().values())
    tem_exp = 0
    l = len(tem_dic_keys)
    for i in range(l):
        if tell_signal(tem_dic_keys[i]) is 'minus':
            tem_exp += -tem_dic_values[i]
        elif tell_signal(tem_dic_keys[i]) is 'plus':
            tem_exp += tem_dic_values[i]
    return tem_exp

def eigensolver(s, hoff, hdiag):
    S = np.array([[1, s.real],[s.real,1]])
    H = np.array([[hdiag.real, hoff.real],[hoff.real, hdiag.real]])
    eigvals = eigh(H, S, eigvals_only=True, subset_by_index=[0, 1])
    return eigvals

def generate_noise_model(prob_1, prob_2):

    noise_model = NoiseModel()

    error_1_dep = noise.depolarizing_error(prob_1, 1)
    error_2_dep = noise.depolarizing_error(prob_2, 2)

    error_1_amp  = noise.amplitude_damping_error(prob_1, 1)
    error_2_amp_tmp = noise.amplitude_damping_error(prob_2, 1)
    error_2_amp = error_2_amp_tmp.expand(error_2_amp_tmp)

    error_1_phase = noise.phase_damping_error(prob_1, 1)
    error_2_phase_tmp = noise.phase_damping_error(prob_2, 1)
    error_2_phase = error_2_phase_tmp.expand(error_2_phase_tmp)

    noise_model.add_all_qubit_quantum_error(error_1_dep, ['u3'])
    noise_model.add_all_qubit_quantum_error(error_2_dep, ['cx'])
    noise_model.add_all_qubit_quantum_error(error_1_amp, ['u3'])
    noise_model.add_all_qubit_quantum_error(error_2_amp, ['cx'])
    noise_model.add_all_qubit_quantum_error(error_1_phase, ['u3'])
    noise_model.add_all_qubit_quantum_error(error_2_phase, ['cx'])
    return noise_model

exp_factory = zne.inference.ExpFactory(scale_factors=[1, 1.5, 2, 2.5, 3, 3.5], asymptote=0.0, avoid_log=True)

def get_data(file_path = "data", noise_scale = 1, num_shots = 10, num_samples = 10):

    prob_1 = 0.00003 * noise_scale
    prob_2 = 0.0015 * noise_scale
    noise_model = generate_noise_model(prob_1 = prob_1, prob_2 = prob_2)

    noise_scaling_function = partial(
    zne.scaling.fold_gates_at_random,
    fidelities = {"single": 1 - prob_1, 'double': 1- prob_2}
    )
    
    def ibmq_executor(circuit):
        result = execute(
            experiments=circuit,
            backend=backend,
            optimization_level=0,# Important to preserve folded gates.
            noise_model=noise_model,
            shots=num_shots,
        ).result()
        tem_exp = tell_exp(result)
        return tem_exp/num_shots
    
    def get_s(circuit, n, num_to_average, fit, noise_scaling_function, plot: False):
    
        # s offdiagonal element 
        overlap_unmitigated_list = []
        overlap_mitigated_list = []
        scaled_noise_expectation_list = []
        c = 0
        while c < n:
            try:
                overlap_unmitigated_list.append(ibmq_executor(circuit))
                overlap_mitigated_list.append(zne.execute_with_zne(circuit, ibmq_executor, num_to_average=num_to_average, factory=fit, scale_noise = noise_scaling_function))
                scaled_noise_expectation_list.append(fit.get_expectation_values())
                c += 1
            except:
                print(c, "get_s failed", datetime.now().strftime("%H:%M:%S"))
        if plot == True:
            fit.plot_fit()
        
        return overlap_unmitigated_list, overlap_mitigated_list

    def get_h_diag(circuit_list, n, num_to_average, fit, noise_scaling_function):

        # H diagonal
        final_unmitigated_list = []
        final_mitigated_list = []
        l =len(circuit_list)
        for c in range(n):
            pauli_unmitigated_list =[]
            pauli_mitigated_list = []
            for i in range(l):
                unmitigated = ibmq_executor(circuit_list[i])
                success = False
                while not success:
                    try:
                        mitigated =  zne.execute_with_zne(circuit_list[i], ibmq_executor, num_to_average=num_to_average, factory = fit, scale_noise = noise_scaling_function)
                        success = True
                    except:
                        print(c, "get_h_diag failed", datetime.now().strftime("%H:%M:%S"))
                pauli_unmitigated_list.append(unmitigated)
                pauli_mitigated_list.append(mitigated)
            final_unmitigated = 0
            final_mitigated= 0
            for i in range(l):
                final_unmitigated += pauli_unmitigated_list[i]*float(h_diag_coefficient_list[i])
                final_mitigated += pauli_mitigated_list[i]*float(h_diag_coefficient_list[i])
            final_unmitigated += h_diag_coefficient_list[2]
            final_mitigated += h_diag_coefficient_list[2]
            final_unmitigated_list.append(final_unmitigated)
            final_mitigated_list.append(final_mitigated)
        return final_unmitigated_list, final_mitigated_list

    def get_h_offdiag(circuit_list, n, num_to_average, fit, noise_scaling_function):
        
        final_unmitigated_list = []
        final_mitigated_list = []
        l =len(circuit_list)
        for c in range(n):
            pauli_unmitigated_list =[]
            pauli_mitigated_list = []
            for i in range(l):
                unmitigated = ibmq_executor(circuit_list[i])
                success = False
                while not success:
                    try:
                        mitigated =  zne.execute_with_zne(circuit_list[i], ibmq_executor, num_to_average = num_to_average, factory = fit, scale_noise = noise_scaling_function)
                        success = True
                    except:
                        print(c, "get_h_offdiag failed", datetime.now().strftime("%H:%M:%S"))
                pauli_unmitigated_list.append(unmitigated)
                pauli_mitigated_list.append(mitigated)
            final_unmitigated = 0
            final_mitigated= 0
            for i in range(l):
                final_unmitigated += pauli_unmitigated_list[i]*float(h_off_coefficient_list[i])
                final_mitigated += pauli_mitigated_list[i]*float(h_off_coefficient_list[i])
            final_unmitigated_list.append(final_unmitigated)
            final_mitigated_list.append(final_mitigated)
        return final_unmitigated_list, final_mitigated_list

    s_off, em_s_off = get_s(circuit = qc_s_offdiagonal, n = num_samples, num_to_average=1, fit= exp_factory, noise_scaling_function = noise_scaling_function, plot = False)
    print("S_off Done", datetime.now().strftime("%H:%M:%S"))
    h_diag, em_h_diag = get_h_diag(circuit_list = qc_h_diag_list, n = num_samples, num_to_average=1, fit = exp_factory, noise_scaling_function = noise_scaling_function)
    print("h_diag done", datetime.now().strftime("%H:%M:%S"))
    h_off, em_h_off = get_h_offdiag(circuit_list = qc_h_offdiag_list, n = num_samples, num_to_average=1, fit = exp_factory, noise_scaling_function = noise_scaling_function)
    print("h_off done", datetime.now().strftime("%H:%M:%S"))

    s0 = []
    t1 = []
    em_s0 = []
    em_t1 = []
    removed_index = []
    for i in range(num_samples):
        try:
            tmp_mitigated = eigensolver(em_s_off[i], em_h_off[i], em_h_diag[i])
            tmp_unmitigated = eigensolver(s_off[i], h_off[i], h_diag[i])
            em_s0.append(tmp_mitigated[0])
            em_t1.append(tmp_mitigated[1])
            s0.append(tmp_unmitigated[0])
            t1.append(tmp_unmitigated[1])
            print(i, datetime.now().strftime("%H:%M:%S"), "success")
        except:
            print(i, datetime.now().strftime("%H:%M:%S"), "failed")
            removed_index.append(i)

    print("Data collection done", datetime.now().strftime("%H:%M:%S"))
    print(f"Removed index: {removed_index}")
    s_off = [val for idx, val in enumerate(s_off) if idx not in removed_index]
    h_off = [val for idx, val in enumerate(h_off) if idx not in removed_index]
    h_diag = [val for idx, val in enumerate(h_diag) if idx not in removed_index]
    em_s_off = [val for idx, val in enumerate(em_s_off) if idx not in removed_index]
    em_h_off = [val for idx, val in enumerate(em_h_off) if idx not in removed_index]
    em_h_diag = [val for idx, val in enumerate(em_h_diag) if idx not in removed_index]

    hadamard_unmitigated = pd.DataFrame({'s_off': s_off, 'h_off': h_off, 'h_diag': h_diag, 's0': s0, 't1': t1})
    hadamard_mitigated = pd.DataFrame({'s_off': em_s_off, 'h_off': em_h_off, 'h_diag': em_h_diag, 's0': em_s0, 't1': em_t1})
    params = f"_L{noise_scale}_S{int(np.log10(num_shots))}.csv"
    hadamard_unmitigated.to_csv(f"{file_path}/hadamard_unmitigated_{params}")
    hadamard_mitigated.to_csv(f"{file_path}/hadamard_mitigated_{params}")

    print(f"noise_scale = {noise_scale}, num_shots = {num_shots}, num_samples = {num_samples}")
    print(f"s_off = {np.mean(s_off):.4f}, h_off = {np.mean(h_off):.4f}, h_diag = {np.mean(h_diag):.4f}, s0 = {np.mean(s0):.4f}, t1 = {np.mean(t1):.4f}")
    print(f"em_s_off = {np.mean(em_s_off):.4f}, em_h_off = {np.mean(em_h_off):.4f}, em_h_diag = {np.mean(em_h_diag):.4f}, em_s0 = {np.mean(em_s0):.4f}, em_t1 = {np.mean(em_t1):.4f}")
    