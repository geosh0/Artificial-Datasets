import sys
import os
import pandas as pd

# 1. Ensure custom models (RLAC, MDH) can be found if they are in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# 2. Import from our newly created src files
from src.data_loader import load_aggregation, load_cluto, load_dartboard, load_pathbased
from src.preprocessing import preprocess_artificial_data
from src.experiments import run_baselines, run_custom_models

def evaluate_artificial_dataset(dataset_name, X, y, n_clusters, rlac_params, mdh_config):
    """Orchestrates the preprocessing and modeling for a single dataset."""
    print(f"\n{'='*60}\nEvaluating: {dataset_name} (k={n_clusters})\n{'='*60}")
    
    # Preprocess (Scale & Encode)
    X_scaled, y_encoded = preprocess_artificial_data(X, y)
    
    # Run Baselines (from experiments.py)
    baseline_res = run_baselines(X_scaled, y_encoded, n_clusters)
    
    # Run Custom Models (from experiments.py)
    custom_res = run_custom_models(X_scaled, y_encoded, n_clusters, rlac_params, mdh_config)
    
    # Combine and Format Results
    all_res = baseline_res + custom_res
    df = pd.DataFrame(all_res, columns=['Model_Type', 'Algorithm', 'Params', 'AMI', 'ARI', 'Silhouette'])
    df.insert(0, 'Dataset', dataset_name) # Add dataset name as the first column
    
    return df

if __name__ == "__main__":
    
    # --- 1. CONFIGURATIONS ---
    rlac_params = {
        'random_state': [42, 43], 
        'bw_adjust':[0.1, 0.2], 
        'r': [None, 200]
    }
    
    mdh_config = {
        "h_multiplier": 1.0, 
        "alphamax_val": 0.9, 
        "alpha_steps": 5, 
        "random_state": 42, 
        "verbose": False, 
        "plot": False
    }

    # --- 2. DATASET REGISTRY ---
    # Tuple format: (Name, Filepath, Load_Function, Target_K)
    datasets =[
        ("Aggregation", r"data/aggregation.arff", load_aggregation, 7),
        ("Cluto-t4-8k", r"data/cluto-t4-8k .arff", load_cluto, 7),
        ("Dartboard1",  r"data/dartboard1 .arff", load_dartboard, 4),
        ("Pathbased",   r"data/pathbased .arff", load_pathbased, 3)
    ]
    
    final_master_results = pd.DataFrame()

    # --- 3. EXECUTION LOOP ---
    for name, path, load_func, k in datasets:
        try:
            # Load the data
            X, y = load_func(path)
            
            # Run the evaluation pipeline
            df_res = evaluate_artificial_dataset(name, X, y, k, rlac_params, mdh_config)
            
            # Append to master table
            final_master_results = pd.concat([final_master_results, df_res], ignore_index=True)
            
        except FileNotFoundError:
            print(f"\n[ERROR] File not found: {path}. Skipping {name}...")
        except Exception as e:
            print(f"\n[ERROR] Failed to process {name}: {e}")

    # --- 4. DISPLAY FINAL TABLE ---
    if not final_master_results.empty:
        print("\n\n" + "="*90)
        print("MASTER BENCHMARK RESULTS (SORTED BY DATASET & AMI)")
        print("="*90)
        
        # Sort so the best model for each dataset is at the top of its section
        final_master_results = final_master_results.sort_values(by=['Dataset', 'AMI'], ascending=[True, False])
        
        # Print without the index numbers for a cleaner look
        print(final_master_results.to_string(index=False))