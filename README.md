# 🚀 PennyLane Quantum MNIST Classifier
## *100% Accuracy with 85 Trainable Parameter

## 🎯 Project Overview

This project demonstrates a **hybrid quantum-classical neural network** built with **PennyLane** and **PyTorch** that achieves **100% test accuracy** on binary classification of MNIST digits (0 vs 1) using only **85 trainable parameters**.

### 📊 **Training Results - 100% Accuracy Achieved**

| Epoch | Loss | Accuracy | Time | Milestone |
|:-----:|:----:|:--------:|:----:|:---------:|
| 1/30 | 0.6871 | 58.3% | 41.7s | |
| 2/30 | 0.6703 | 77.8% | 32.8s | |
| 3/30 | 0.6472 | 86.1% | 34.1s | |
| 4/30 | 0.6169 | 88.9% | 34.3s | |
| 5/30 | 0.5725 | 94.4% | 33.8s | 🎯 Baseline Surpassed |
| 6/30 | 0.5143 | 97.2% | 33.2s | |
| 7/30 | 0.4417 | 91.7% | 33.5s | |
| 8/30 | 0.3724 | 93.1% | 33.8s | |
| 9/30 | 0.3094 | 94.4% | 33.8s | |
| 10/30 | 0.2638 | 97.2% | 32.8s | |
| 11/30 | 0.2266 | 98.6% | 33.8s | ⚡ Near-Perfect |
| 12/30 | 0.1912 | 97.2% | 33.7s | |
| 13/30 | 0.1705 | 98.6% | 34.0s | |
| 14/30 | 0.1529 | 98.6% | 33.0s | |
| 15/30 | 0.1282 | 98.6% | 33.6s | |
| 16/30 | 0.1166 | 98.6% | 32.9s | |
| 17/30 | 0.1046 | 98.6% | 34.1s | |
| **18/30** | **0.0940** | **100.0%** | **32.8s** | 🏆 **PERFECT!** |
| 19/30 | 0.0834 | 100.0% | 33.9s | |
| 20/30 | 0.0771 | 100.0% | 33.8s | |
| 21/30 | 0.0708 | 100.0% | 33.8s | |
| 22/30 | 0.0683 | 100.0% | 33.1s | |
| 23/30 | 0.0631 | 100.0% | 34.2s | |
| 24/30 | 0.0528 | 100.0% | 33.7s | |
| 25/30 | 0.0477 | 100.0% | 34.2s | |
| 26/30 | 0.0468 | 100.0% | 33.1s | |
| 27/30 | 0.0428 | 100.0% | 34.0s | |
| 28/30 | 0.0412 | 100.0% | 33.7s | |
| 29/30 | 0.0352 | 100.0% | 33.8s | |
| 30/30 | 0.0340 | 100.0% | 33.5s | |

---

### 📈 **Performance Summary**

| Metric | Value |
|--------|-------|
| 🏆 **Best Accuracy** | **100.0%** |
| 📉 **Final Loss** | 0.0340 |
| ⏱️ **Convergence Epoch** | 18 |
| 📊 **Total Training Time** | ~16.5 minutes |
| 🧮 **Parameters** | 85 |
| 💫 **Qubits** | 4 |

---

### 📉 **Loss Convergence Analysis**

| Stage | Epoch | Loss | Improvement |
|-------|-------|------|-------------|
| Initial | 1 | 0.6871 | - |
| Mid | 10 | 0.2638 | ↓ 61.6% |
| Near-Perfect | 11 | 0.2266 | ↓ 67.0% |
| **Perfect** | **18** | **0.0940** | ↓ **86.3%** |
| Final | 30 | 0.0340 | ↓ **95.1%** |

---

### ⚡ **Accuracy Progression**

| Phase | Epoch | Accuracy | Status |
|-------|-------|----------|--------|
| Start | 1 | 58.3% | 🔄 Learning |
| Good | 5 | 94.4% | ✅ Baseline |
| High | 10 | 97.2% | 📈 Improving |
| Near-Perfect | 11-17 | 98.6% | ⚡ Almost |
| **Perfect** | **18-30** | **100.0%** | 🏆 **ACHIEVED** |

---

### 🎯 **Key Milestones**

| Epoch | Achievement |
|-------|-------------|
| 5 | Surpassed classical baseline (94.4%) |
| 11 | First near-perfect score (98.6%) |
| **18** | **🏆 100% ACCURACY ACHIEVED** |
| 30 | Maintained 100% for 12 epochs |

---

### 📊 **Final Model Performance**

**🏆 Summary: 100% Accuracy Achieved at Epoch 18 with 85 Trainable Parameters**
### Technical Specifications

| Component | Selection | Rationale |
|-----------|-----------|-----------|
| **Device** | `default.qubit` | Fast simulation, production ready |
| **Embedding** | `AmplitudeEmbedding` | 2ⁿ features → n qubits, max density |
| **Variational** | `StronglyEntanglingLayers` | Max expressivity, proven performance |
| **Depth** | 6 layers | Sufficient complexity, no overfitting |
| **Differentiation** | `adjoint` | 10x faster than parameter-shift |
| **Optimizer** | Adam (LR=0.01) | Adaptive, stable convergence |
| **Batch Size** | 32 | Balanced speed & stability |
| **Interface** | `torch` | Seamless PyTorch integration |

---

## 📉 Muhammad Hasnain
