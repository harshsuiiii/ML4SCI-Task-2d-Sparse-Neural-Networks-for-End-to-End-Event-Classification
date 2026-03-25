## 📌 Problem Statement

This project focuses on **end-to-end event classification** using **sparse neural networks** on high-dimensional scientific data. The goal is to efficiently learn representations from unlabeled data, fine-tune for classification, and analyze the trade-off between computational cost and performance.

---

## 🧠 Approach

Our pipeline consists of the following steps:

1. **Data Handling**

   * Large-scale HDF5 dataset (~38GB) handled using memory-efficient lazy loading.
   * Multi-channel input data reshaped to channel-first format for CNN processing.

2. **Model Architecture**

   * Implemented a **Sparse Residual Convolutional Neural Network** inspired by submanifold sparse convolutions.
   * Compared against a **Dense CNN baseline**.

3. **Training Strategy**

   * Supervised training using binary classification objective.
   * Efficient batching using PyTorch DataLoader.

4. **Model Pruning**

   * Applied **global unstructured pruning** to induce sparsity.
   * Evaluated performance across multiple pruning ratios.

5. **Fine-Tuning**

   * Fine-tuned pruned models to recover performance and stabilize results.

6. **Efficiency Analysis**

   * Computed **effective FLOPS** by scaling theoretical FLOPS with model sparsity.

---

## 🏗️ Models Used

### 🔹 Dense CNN (Baseline)

* Standard convolutional layers
* Used for comparison with sparse models

### 🔹 Sparse ResNet (Proposed)

* Residual architecture
* Sparsity preserved via masking (VSC-inspired)
* Designed for efficient computation on sparse data

---

## 📊 Results & Visualizations

### 📈 Training Loss Curve

* Demonstrates stable convergence during training

### 📉 ROC Curve

* Evaluates classification performance across thresholds

### 📊 Confusion Matrix

* Shows classification distribution and error patterns

### 📉 Error vs FLOPS (Key Result)

* Highlights trade-off between efficiency and performance

### 📈 Accuracy vs Pruning Ratio

* Shows degradation trend with increasing sparsity

### 📊 Dense vs Sparse Comparison

* Demonstrates efficiency gains of sparse models

---

## 🔬 Key Observations

* **Moderate pruning (40–60%) achieves the best trade-off** between efficiency and accuracy.
* Sparse architectures significantly reduce redundant computations in high-dimensional data.
* Fine-tuning after pruning is critical to recover model performance.
* Excessive pruning (>80%) leads to sharp performance degradation due to loss of important features.

---

## 🧠 Technical Insights

* Unstructured pruning does not physically remove parameters; hence, **effective FLOPS** is estimated using sparsity scaling.
* Sparse convolutional design aligns well with the inherent sparsity of scientific datasets.
* Residual connections improve stability in sparse architectures.

---

## 🏁 Conclusion

This work demonstrates that **sparse neural networks combined with pruning** can significantly reduce computational cost while maintaining strong predictive performance. The approach is particularly well-suited for large-scale scientific datasets where efficiency is critical.

---

## ⚙️ Tech Stack

* Python
* PyTorch
* HDF5
* NumPy
* Matplotlib / Seaborn
* ptflops

---

## 📂 Repository Structure

```
├── models/            # Model architectures
├── training/          # Training & pruning scripts
├── evaluation/        # Metrics & plotting
├── notebooks/         # Final notebook (main submission)
├── results/           # Generated plots
├── README.md
```

---

## 🚀 How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the notebook:

```
notebooks/final_submission.ipynb
```

---

## 🙌 Acknowledgements

* ML4SCI Program
* PyTorch & scientific ML community

---

## 📌 Author

**Harsh Vardhan Singh**
