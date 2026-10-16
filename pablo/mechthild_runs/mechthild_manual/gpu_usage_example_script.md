"""
This file isn't used. it's just an example script for managing the HPC, and showing its features.
"""

import os
import subprocess
import glob
import time
import datetime
import argparse
import json
--- CONFIGURATION ---

JOB_FILE = "slurm_jobs.json"
QUEUE_DIR = "data/queue_state"
RESULTS_DIR = "data/results"
LOG_DIR = "logs"
EMAIL = "griehl@mpi-magdeburg.mpg.de"
PARTITIONS (Check your cluster documentation!)

PARTITION_GPU = "gpu"
PARTITION_CPU = "long"  # Usually 'cpu', 'batch', 'normal', or 'standard'
Hardware Limits

CAPACITY = {
    "H100": 28,
    "A100": 12,
    "CPU": 50  # Arbitrary high limit for CPU jobs since they are cheap
}
Resource Profiles

RESOURCE_PROFILES = {
    "H100": {"cpus": 16, "mem": "64G", "threads": 4},
    "A100": {"cpus": 8, "mem": "32G", "threads": 2},
    "CPU": {"cpus": 4, "mem": "16G", "threads": 4}
}

TIME_LIMIT = "12:00:00"

class Colors:
    HEADER = '\033[95m';
    BLUE = '\033[94m';
    GREEN = '\033[92m'
    WARNING = '\033[93m';
    FAIL = '\033[91m';
    ENDC = '\033[0m';
    BOLD = '\033[1m'

def log(msg, type="INFO"):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    color = Colors.BLUE if type == "INFO" else (Colors.WARNING if type == "WARN" else Colors.GREEN)
    print(f"{Colors.BOLD}[{timestamp}] {color}{type}:{Colors.ENDC} {msg}")

def get_queue_counts():
    if not os.path.exists(QUEUE_DIR): return None
    pending = len(glob.glob(os.path.join(QUEUE_DIR, 'pending', '.lock')))
    running = len(glob.glob(os.path.join(QUEUE_DIR, 'running', '.lock')))
    done = len(glob.glob(os.path.join(QUEUE_DIR, 'done', '*.lock')))
    return {"pending": pending, "running": running, "done": done, "total": pending + running + done}

def get_pending_types():
    if not os.path.exists(JOB_FILE): return {'DeepGP': 0, 'StandardGP': 0}
    with open(JOB_FILE, 'r') as f:
        jobs = json.load(f)
    pending = glob.glob(os.path.join(QUEUE_DIR, 'pending', '*.lock'))
    counts = {'CustomDeepGP': 0, 'StandardGP': 0}
    for p in pending:
        try:
            idx = int(os.path.basename(p).split('_')[1].split('.')[0])
            method = jobs[idx]['method']
            if method in counts: counts[method] += 1
        except:
            pass
    return counts

def get_active_workers(prefix):
    user = os.environ.get('USER')
    try:
        res = subprocess.check_output(["squeue", "-u", user, "--noheader", "--format=%j"], text=True)
        return len([l for l in res.split('\n') if prefix in l])
    except:
        return 0

def submit_array(hardware_type, num_workers, job_prefix):
    if num_workers <= 0: return
    profile = RESOURCE_PROFILES[hardware_type]
    partition = PARTITION_GPU if hardware_type in ["H100", "A100"] else PARTITION_CPU
    job_name = f"{job_prefix}_{hardware_type}"
# GPU Arg Logic
gpu_arg = f"#SBATCH --gpus={hardware_type}:1" if hardware_type in ["H100", "A100"] else ""

script = f"""#!/bin/bash

#SBATCH -J {job_name}

#SBATCH --array=1-{num_workers}

#SBATCH --partition={partition}
{gpu_arg}

#SBATCH --cpus-per-task={profile['cpus']}

#SBATCH --mem={profile['mem']}

#SBATCH --time={TIME_LIMIT}

#SBATCH -o {LOG_DIR}/{job_name}%A%a.out

#SBATCH -e {LOG_DIR}/{job_name}%A%a.err

#SBATCH --mail-type=FAIL

#SBATCH --mail-user={EMAIL}

source venv/bin/activate
export OMP_NUM_THREADS={profile['threads']}
export MKL_NUM_THREADS={profile['threads']}
export TORCH_NUM_THREADS={profile['threads']}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

sleep $((RANDOM % 10))

python hpc_manager.py worker \
    --job_file {JOB_FILE} \
    --queue_dir {QUEUE_DIR} \
    --save_dir {RESULTS_DIR} \
    --hardware {hardware_type}
"""
    fname = f".submit_{hardware_type}.sh"
    with open(fname, "w") as f:
        f.write(script)
log(f"Submitting {num_workers} {hardware_type} workers...", "INFO")
try:
    subprocess.run(["sbatch", fname], check=True)
    log(f"Launched {hardware_type} array.", "SUCCESS")
except:
    log(f"Failed to submit {hardware_type}.", "FAIL")
if os.path.exists(fname): os.remove(fname)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--job_prefix", default="Exp")
    args = parser.parse_args()
os.makedirs(LOG_DIR, exist_ok=True)
if not os.path.exists(QUEUE_DIR): return

while True:
    counts = get_queue_counts()
    pending_types = get_pending_types()

    # 1. GPU Logic (DeepGP)
    n_deep = pending_types['CustomDeepGP']
    if n_deep > 0:
        active_h100 = get_active_workers(f"{args.job_prefix}_H100")
        active_a100 = get_active_workers(f"{args.job_prefix}_A100")

        free_h100 = max(0, CAPACITY['H100'] - active_h100)
        free_a100 = max(0, CAPACITY['A100'] - active_a100)

        take_h100 = min(n_deep, free_h100)
        n_deep -= take_h100
        take_a100 = min(n_deep, free_a100)

        submit_array("H100", take_h100, args.job_prefix)
        submit_array("A100", take_a100, args.job_prefix)

    # 2. CPU Logic (StandardGP)
    n_std = pending_types['StandardGP']
    if n_std > 0:
        active_cpu = get_active_workers(f"{args.job_prefix}_CPU")
        free_cpu = max(0, CAPACITY['CPU'] - active_cpu)
        take_cpu = min(n_std, free_cpu)
        submit_array("CPU", take_cpu, args.job_prefix)

    # Monitor UI
    print("\033[H\033[J", end="")
    print(f"Queue: {counts['pending']} Pending | {counts['running']} Running | {counts['done']} Done")
    print(f"Pending Breakdown: DeepGP={pending_types['CustomDeepGP']}, StdGP={pending_types['StandardGP']}")

    if not args.monitor or (counts['pending'] == 0 and counts['running'] == 0):
        break
    time.sleep(30)

if name == "main":
    main()
