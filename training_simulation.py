# 📊 AI Training Dashboard Simulator

### 🚀 Overview
This project is a **simulation tool** designed to visualize the internal mechanics of Distributed AI Training. 

For Engineering Managers and Architects, understanding the **Training Lifecycle** is critical for cost estimation and resource planning. This tool simulates the relationship between **Epochs** and **Loss Convergence**, as well as real-time **GPU Cluster Utilization**.

### 🧠 Key Concepts Demonstrated
1.  **Loss Convergence:** Visualizes how the model's error rate decreases over time (Epochs) as it learns from the dataset.
2.  **Distributed Data Parallelism (DDP):** Simulates a multi-GPU setup where the dataset is chunked and processed in parallel to reduce training time.
3.  **Resource Monitoring:** Mimics live GPU utilization metrics (85%-100% load) typical of high-performance computing (HPC) clusters.

### 🛠️ Tech Stack
* **Language:** Python 3.9
* **Visualization:** Matplotlib
* **Computation:** NumPy
* **Environment:** Optimized for Jupyter Notebooks / Google Colab

### 📉 How to Interpret the Graphs
* **Left Graph (Loss Curve):** A steep drop indicates rapid learning. A flat line at the end indicates "Convergence" (the model has stopped learning), signaling that training should be stopped to save costs.
* **Right Graph (GPU Usage):** Shows the load across 4 simulated GPUs. In a healthy distributed system, usage should remain high and balanced across all nodes.

### 🏃‍♂️ How to Run
This code is best viewed in a Jupyter Notebook or Google Colab environment to see the live animation.

```bash
# Install dependencies
pip install -r requirements.txt

# Run the simulation (if using script)
python training_simulation.py