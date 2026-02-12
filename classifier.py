import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time

# ============ CONFIGURATION ============
DEVICE_NAME = "default.qubit"
N_QUBITS = 4
N_LAYERS = 6
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.01
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
start_global = time.time()

print("⚡ FIXED QUANTUM CLASSIFIER ⚡")
print("=" * 60)

# ============ FIXED DATA PREP - NOW 2^N_QUBITS FEATURES! ============
print("\n📊 DATA PREPARATION")
print("-" * 60)

digits = load_digits()
X, y = digits.data, digits.target
mask = (y == 0) | (y == 1)
X, y = X[mask], y[mask]

# CRITICAL FIX: For amplitude embedding, we need 2^N_QUBITS features
N_FEATURES = 2**N_QUBITS  # This is the key fix!

print(f"   Original features: {X.shape[1]}")
print(f"   Target features for amplitude embedding: {N_FEATURES} (2^{N_QUBITS})")

# PCA to exactly 2^N_QUBITS features
pca = PCA(n_components=N_FEATURES)
X_pca = pca.fit_transform(X)

# Normalize for amplitude embedding (must have unit norm)
from sklearn.preprocessing import Normalizer
normalizer = Normalizer(norm='l2')
X_normalized = normalizer.fit_transform(X_pca)

X_tensor = torch.tensor(X_normalized, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=SEED
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

print(f"✅ Final data shape: {X_train.shape[1]} features for {N_QUBITS} qubits")
print(f"✅ {len(X_train)} train, {len(X_test)} test samples")

# ============ FIXED QUANTUM CIRCUIT ============
print("\n🔬 QUANTUM CIRCUIT")
print("-" * 60)

dev = qml.device(DEVICE_NAME, wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_circuit(inputs, weights):
    """
    FIXED: inputs has shape (16,) for 4 qubits (2^4 = 16)
    """
    # Amplitude embedding with CORRECT dimensions
    qml.AmplitudeEmbedding(features=inputs, wires=range(N_QUBITS), normalize=True)
    
    # Strongly entangling layers - better than BasicEntangler for learning
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    
    return qml.expval(qml.PauliZ(0))

# Visualize circuit
sample_input = torch.randn(N_FEATURES)
sample_input = sample_input / torch.norm(sample_input)  # Normalize
sample_weights = torch.randn(N_LAYERS, N_QUBITS, 3)

fig, ax = qml.draw_mpl(quantum_circuit)(sample_input, sample_weights)
fig.suptitle(f'Quantum Circuit: {N_QUBITS} Qubits, {N_LAYERS} Layers', 
             fontsize=14, fontweight='bold')
plt.savefig('quantum_circuit_fixed.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Circuit diagram saved")

# ============ FIXED MODEL ============
class FixedQuantumClassifier(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_features = 2**n_qubits  # CRITICAL FIX
        
        # Quantum layer with correct weight shapes for StronglyEntanglingLayers
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x already has correct shape (batch_size, 2^n_qubits)
        batch_size = x.shape[0]
        
        # Process batch
        quantum_outputs = []
        for i in range(batch_size):
            q_out = self.qlayer(x[i])
            quantum_outputs.append(q_out)
        
        x = torch.stack(quantum_outputs).reshape(-1, 1)
        return self.output(x)

# Initialize model
model = FixedQuantumClassifier(N_QUBITS, N_LAYERS)
print(f"\n✅ Model created: {sum(p.numel() for p in model.parameters())} trainable parameters")

# ============ FIXED TRAINING ============
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n🚀 TRAINING STARTED (should learn to 90%+ now!)")
print("-" * 60)

train_losses = []
test_accuracies = []
best_acc = 0

for epoch in range(EPOCHS):
    # Training
    model.train()
    total_loss = 0
    epoch_start = time.time()
    
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predicted = (output > 0.5).float()
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / len(X_test)
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    test_accuracies.append(accuracy)
    
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), 'quantum_model_fixed.pt')
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {accuracy:.1f}% | Time: {time.time()-epoch_start:.1f}s")

print("-" * 60)
print(f"🏆 Best accuracy: {best_acc:.1f}%")

# ============ OPTIONAL: SWITCH TO ANGLE EMBEDDING (FASTER) ============
print("\n🔄 QUICK COMPARISON: Angle Embedding Version")
print("-" * 60)

# Alternative circuit using Angle Embedding (n_qubits features, faster)
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def angle_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))

class AngleQuantumClassifier(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        
        # Simple preprocessing to n_qubits
        self.preprocess = nn.Linear(n_qubits, n_qubits)
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(angle_circuit, weight_shapes)
        
        self.output = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Reduce to n_qubits features
        x = self.preprocess(x[:, :self.n_qubits])
        batch_size = x.shape[0]
        
        quantum_outputs = []
        for i in range(batch_size):
            q_out = self.qlayer(x[i])
            quantum_outputs.append(q_out)
        
        x = torch.stack(quantum_outputs).reshape(-1, 1)
        return self.output(x)

print("\n✅ Use Angle Embedding if you want faster training with 4 features")
print("✅ Use Amplitude Embedding if you want maximum expressivity with 16 features")

# ============ FINAL VISUALIZATION ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, 'b-', linewidth=2)
ax1.set_title('Training Loss (Amplitude Embedding)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(test_accuracies, 'g-', linewidth=2)
ax2.set_title(f'Test Accuracy: {best_acc:.1f}%', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quantum_results_fixed.png', dpi=150)
plt.show()

total_time = time.time() - start_global
print("\n" + "=" * 60)
print(f"✨ COMPLETE! Training time: {total_time:.1f} seconds")
print("📁 Files saved:")
print("   • quantum_circuit_fixed.png")
print("   • quantum_results_fixed.png")
print("   • quantum_model_fixed.pt")
print("=" * 60)
