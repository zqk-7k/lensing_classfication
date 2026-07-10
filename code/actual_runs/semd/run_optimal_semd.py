import os, subprocess, re, time

# ================= 1. Environment & Path Settings =================
WORK_DIR = "/root/autodl-tmp/wjx_project/classcify-gw-lensing-pairs-main/SEMD-LensedGW-main/"
os.chdir(WORK_DIR)

GPU_ID = "0"
MAX_EPOCHS = 300
MAIN_FILE = "main.py"

# ================= 2. Optimal LR Configurations =================
# Format: ("Model", "Mode", Optimal_LR)
TASKS = [
    ("PM", "pure", 5e-5),
    ("PM", "noisy", 1e-5),
    ("SIS", "pure", 5e-5),
    ("SIS", "noisy", 5e-5)
]

CFG = "config.py"
LOG_DIR = "./optimal_logs"
HISTORY_FILE = "optimal_history_300ep.txt"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

with open(HISTORY_FILE, "a", encoding='utf-8') as f:
    f.write(f"\n\n{'#'*30}\nOptimal Run Started: {time.ctime()}\nTotal Tasks: {len(TASKS)}\nEnvironment: {MAIN_FILE}, GPU {GPU_ID}, {MAX_EPOCHS} Epochs\n{'#'*30}\n")

def update_cfg(m, d):
    if not os.path.exists(CFG):
        print(f"[Warning] Config file {CFG} not found. Skipping rewrite.")
        return

    with open(CFG, 'r', encoding='utf-8') as f:
        content = f.read()

    content = re.sub(r'MODEL_NAME\s*=\s*[\'"].*?[\'"]', f"MODEL_NAME = '{m}'", content)
    content = re.sub(r'DATA_MODE\s*=\s*[\'"].*?[\'"]', f"DATA_MODE = '{d}'", content)

    with open(CFG, 'w', encoding='utf-8') as f:
        f.write(content)

def run_train(m, d, lr):
    log_file = os.path.join(LOG_DIR, f"{m}_{d}_lr{lr}_ep{MAX_EPOCHS}.log")

    # [FIXED] Removed --model and --mode, kept only --lr and --epochs
    cmd = f"CUDA_VISIBLE_DEVICES={GPU_ID} python {MAIN_FILE} --lr {lr} --epochs {MAX_EPOCHS}"

    print(f"\n[Task Started] Model: {m} | Mode: {d} | LR: {lr} | Target: {MAX_EPOCHS} Ep")
    print(f"Executing: {cmd}")

    with open(HISTORY_FILE, "a", encoding='utf-8') as hf, open(log_file, "w") as lf:
        hf.write(f"\n[Task] Model:{m} | Mode:{d} | LR:{lr} | Target:{MAX_EPOCHS} Ep\n")
        hf.write("-" * 60 + "\n\n")

        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        best_acc = 0.0

        for line in process.stdout:
            lf.write(line)
            lf.flush()

            acc_matches = re.findall(r'[Aa]cc[=:]\s*([0-9.]+)', line)
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
                hf.flush()

        process.wait()

    return best_acc

if __name__ == "__main__":
    summary = []
    start_time = time.time()

    for model_name, data_mode, lr in TASKS:
        # update_cfg handles the model and mode switching implicitly!
        update_cfg(model_name, data_mode)
        max_acc = run_train(model_name, data_mode, lr)

        summary.append({
            "Model": model_name,
            "Mode": data_mode,
            "LR": lr,
            "Best_Acc": max_acc
        })

        finish_msg = f"\n[Task Finished] {model_name}-{data_mode}-LR:{lr} | Best Acc: {max_acc:.4f}\n"
        with open(HISTORY_FILE, "a", encoding='utf-8') as hf:
            hf.write(finish_msg)
        print(finish_msg)

    print("\n" + "="*50)
    print("All optimal tasks completed!")
    print(f"Total time elapsed: {(time.time() - start_time)/3600:.2f} hours")
    print("Summary:")
    for s in summary:
        print(f" - {s['Model']} ({s['Mode']}) LR: {s['LR']} -> Best Acc: {s['Best_Acc']:.4f}")
    print("="*50 + "\n")