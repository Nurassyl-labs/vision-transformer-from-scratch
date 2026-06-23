import subprocess
import os
import sys

def run_command(command):
    print(f"Executing command: {command}")
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    process = subprocess.Popen(
        command,
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
#Print output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            sys.stdout.flush()
            
    rc = process.poll()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, command)
    return rc

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working Directory: {os.getcwd()}")
    
    # 1. Run Baseline (CLS Token) experiment
    print("1/3 Running Baseline Training (Class Token)")
    run_command("python3 train.py --epochs 15 --exp_name baseline")
    
    # 2. Run Ablation (GAP) experiment
    print("\n2/3 Running Ablation Training (Global Average Pooling)")
    run_command("python3 train.py --epochs 15 --use_gap --exp_name ablation")
    
    # 3. Generate plots
    print("\n3/3 Generating Comparison Plots")
    run_command("python3 plot.py")
    
    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()
