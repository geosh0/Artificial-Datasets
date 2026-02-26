# 🧪 Artificial Datasets Clustering Benchmark

This repository provides a standardized benchmarking environment for evaluating clustering models on complex, 2D artificial datasets. These datasets are specifically designed to test an algorithm's ability to handle non-convex shapes, varying densities, and spatial noise—geometric challenges where standard algorithms often fail.

As our baseline, we utilize standard algorithms such as K-Means, Spectral Clustering, and Agglomerative Clustering. We then benchmark these against two specialized, continuous-optimization models: **RLAC** and **MDH**.

## 🧠 Custom Thesis Models Evaluated
Alongside the baselines, we evaluate two specialized clustering models (hosted in their own standalone repositories):
* **[RLAC](#)** (Random Line Approximation Clustering): The implementation and evaluation of this model served as the primary focus of my academic thesis.
* **[MDH](#)** (Minimum Density Hyperplanes): Integrated concurrently to broaden the scope of the comparative analysis, providing a more comprehensive evaluation of hyperplane-based clustering results.

## 📊 Datasets
We utilize classic 2D artificial datasets, which allow for clear visual and mathematical interpretation of how models handle complex spatial distributions.
* **Aggregation** ($k=7$)
* **Cluto-t4-8k** ($k=7$, noise filtered)
* **Dartboard1** ($k=4$)
* **Pathbased** ($k=3$)

*Data Source:* [Clustering Benchmark Repository by deric](https://github.com/deric/clustering-benchmark/tree/master/src/main/resources/datasets/artificial)

## ⚙️ Pipeline Architecture
Instead of using repetitive procedural scripts, this repository is built on a strictly modular **DRY (Don't Repeat Yourself)** architecture:
* **`src/data_loader.py`**: Features a unified `_load_arff_to_df` abstraction that cleanly parses `.arff` files, handling dataset-specific edge cases (like filtering string 'noise' tags) without cluttering the main logic.
* **`src/preprocessing.py`**: Standardizes features and encodes labels while intentionally preserving the original geometric structures (no generic outlier removal is applied to these shape-based datasets).
* **`src/experiments.py`**: A universal evaluation engine that runs the models and calculates **AMI**, **ARI**, and **Silhouette Scores**.
* **`main.py`**: A clean execution loop utilizing a Dataset Registry pattern to process all files and generate a final benchmark table.

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/geosh0/artificial-dataset-processing.git
   cd artificial-dataset-processing

## Install dependencies:
```Bash
pip install pandas numpy scikit-learn liac-arff
```
> (Ensure you also have the dependencies required by RLAC/MDH).

## Run the Master Pipeline:
```Bash
python main.py
```
* The script will automatically parse the .arff files, scale the features, apply the models, and print a consolidated Pandas DataFrame comparing all configurations sorted by AMI.
