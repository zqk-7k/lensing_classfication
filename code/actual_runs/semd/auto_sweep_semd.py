# -*- coding: utf-8 -*-
import os, subprocess, re, time
import pandas as pd

# ================= SEMD Auto Sweep Config =================
TASKS = [
    ("SIS", "pure"),
    ("SIS", "noisy"),
    ("PM", "pure"),
    ("PM", "noisy")
]

LR_RANGE = [1e-5, 3e-5, 5e-5, 8e-5, 1e-4]
MAX_EPOCHS = 50

CFG = "config.py"
LOG_DIR = "./sweep_logs"
HISTORY_FILE = "total_history.txt"

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

with open(HISTORY_FILE, "a", encoding='utf-8') as f:
    f.write(f"\n\n{'#'*30}\nAuto Sweep Started: {time.ctime()}\nTotal Tasks: {len(TASKS)*len(LR_RANGE)}\nEnvironment: SEMD (main.py, GPU 0)\n{'#'*30}\n")

def update_cfg(m, d):
    if not os.path.exists(CFG):
        print(f"[Warning] Config file {CFG} not found. Skipping rewrite.")
        return

    with open(CFG, 'r', encoding='utf-8') as f:
        content = f.read()

    content = re.sub(r'MODEL_TYPE\s*=\s*["\'].*?["\']', f'MODEL_TYPE = "{m}"', content)
    content = re.sub(r'DATA_MODE\s*=\s*["\'].*?["\']', f'DATA_MODE = "{d}"', content)

    with open(CFG, 'w', encoding='utf-8') as f:
        f.write(content)

def run_train(m, d, lr):
    sub_log = f"{LOG_DIR}/{m}_{d}_{lr}.log"
    header = f"\n[Task] Model:{m} | Mode:{d} | LR:{lr} | Target:{MAX_EPOCHS} Ep\n{'-'*60}\n"
    print(header)

    with open(HISTORY_FILE, "a", encoding='utf-8') as hf: hf.write(header)

    my_env = os.environ.copy()
    my_env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [
        "python", "main.py",
        "--lr", str(lr),
        "--epochs", str(MAX_EPOCHS)
    ]

    best_acc = 0.0
    with open(sub_log, "w") as f, open(HISTORY_FILE, "a", encoding='utf-8') as hf:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=my_env)
        for line in process.stdout:
            f.write(line)

            acc_matches = re.findall(r'(?:acc|accuracy)\s*[:=]\s*([\d.]+)', line, flags=re.IGNORECASE)

            if acc_matches:
                cur_acc = float(acc_matches[-1])
                if cur_acc > best_acc and cur_acc > 0:
                    best_acc = cur_acc
                    tagged_line = f"  {line.strip()}  <-- [Saved Best]\n"
                else:
                    tagged_line = f"  {line.strip()}\n"

                print(tagged_line, end='')
                hf.write(tagged_line)
                hf.flush()
            else:
                hf.write(line)
        process.wait()

    return best_acc

if __name__ == "__main__":
    summary = []
    start_time = time.time()

    for model_name, data_mode in TASKS:
        for lr in LR_RANGE:
            update_cfg(model_name, data_mode)
            max_acc = run_train(model_name, data_mode, lr)

            summary.append({
                "Model": model_name,
                "Mode": data_mode,
                "LR": lr,
                "Best_Acc": max_acc
            })

            finish_msg = f"\n[Task Finished] {model_name}-{data_mode}-LR:{lr} | Best Acc: {max_acc:.4f}\n"
            with open(HISTORY_FILE, "a", encoding='utf-8') as hf: hf.write(finish_msg)
            print(finish_msg)

    pd.DataFrame(summary).to_csv("full_sweep_comparison_report.csv", index=False)
    print(f"\n{'='*50}\nAll sweep tasks completed!")
    print(f"Total time elapsed: {(time.time()-start_time)/3600:.2f} hours")
    print(f"Report saved to: full_sweep_comparison_report.csv")