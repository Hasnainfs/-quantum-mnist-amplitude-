dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_circuit(inputs, weights):
    # Amplitude Embedding: 16 classical features → 4 qubits
    qml.AmplitudeEmbedding(features=inputs, wires=range(4), normalize=True)
    
    # 6 layers of StronglyEntanglingLayers for maximum expressivity
    qml.StronglyEntanglingLayers(weights, wires=range(4))
    
    # PauliZ measurement on first qubit for binary classification
    return qml.expval(qml.PauliZ(0))
