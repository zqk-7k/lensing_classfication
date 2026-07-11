# -*- coding: utf-8 -*-
import os
import subprocess

# Task List: (Model, Mode)
# It will run SIS_pure, SIS_noisy, PM_pure, PM_noisy sequentially.
experiments = [
    ("SIS", "pure"),
    ("SIS", "noisy"),
    ("PM", "pure"),
    ("PM", "noisy")
]

def update_config(model_type, data_mode):
    """Automatically update config.py and config_preprocess.py"""
    files = ["config.py", "config_preprocess.py"]
    for file_path in files:
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in lines:
                # Update MODEL_TYPE
                if line.strip().startswith("MODEL_TYPE ="):
                    f.write(f'MODEL_TYPE = "{model_type}"\n')
                # Update DATA_MODE
                elif line.strip().startswith("DATA_MODE  ="):
                    f.write(f'DATA_MODE  = "{data_mode}"\n')
                # Force CQT method
                elif line.strip().startswith("TRANSFORM_METHOD ="):
                    f.write(f"TRANSFORM_METHOD = 'cqt'\n")
                # Force safer LR (1e-5) to prevent NaN
                elif line.strip().startswith("LR          ="):
                    f.write(f"LR          = 1e-5\n")
                else:
                    f.write(line)

def run():
    for model, data in experiments:
        print(f"\n?? [TASK START] Model: {model}, Mode: {data}")
        
        # 1. Update Config Files
        update_config(model, data)
        
        # 2. Run Preprocessing (NPY -> PNG)
        print(f"--- Step 1: Preprocessing ---")
        # Run preprocess_offline.py as provided in your files
        subprocess.run(["python", "preprocess_offline.py"], check=True)
        
        # 3. Run Training with main.py (Single GPU mode)
        print(f"--- Step 2: Training with main.py ---")
        env = os.environ.copy()
        # Explicitly use GPU 0 to stay safe
        env["CUDA_VISIBLE_DEVICES"] = "0"
        
        # Args match your main.py: --epochs, --batch_size, --lr
        cmd = [
            "python", "main.py", 
            "--epochs", "300", 
            "--batch_size", "64", 
            "--lr", "1e-5"
        ]
        subprocess.run(cmd, env=env, check=True)
        
        print(f"? [COMPLETED] {model} {data}")
        print("-" * 50)

if __name__ == "__main__":
    run()