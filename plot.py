import os
import json
import matplotlib.pyplot as plt

def load_history(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found!")
        return None
    with open(filepath, "r") as f:
        return json.load(f)

def main():
    os.makedirs("plots", exist_ok=True)
    
    baseline = load_history("logs/baseline_history.json")
    ablation = load_history("logs/ablation_history.json")
    
    if baseline is None or ablation is None:
        print("Could not load log files. Make sure both experiments have finished successfully.")
        return
        
    epochs = list(range(1, len(baseline["train_loss"]) + 1))
    
    # Apply standard grid styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Plotting Loss Curves
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(epochs, baseline["train_loss"], label="Baseline (CLS) - Train Loss", color="#1f77b4", linestyle="--", linewidth=2)
    plt.plot(epochs, baseline["val_loss"], label="Baseline (CLS) - Val Loss", color="#1f77b4", marker="o", linewidth=2)
    
    plt.plot(epochs, ablation["train_loss"], label="Ablation (GAP) - Train Loss", color="#ff7f0e", linestyle="--", linewidth=2)
    plt.plot(epochs, ablation["val_loss"], label="Ablation (GAP) - Val Loss", color="#ff7f0e", marker="s", linewidth=2)
    
    plt.title("Cross-Entropy Loss Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xticks(epochs)
    plt.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    plt.savefig("plots/loss_comparison.png")
    plt.close()
    
    # 2. Plotting Accuracy Curves
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(epochs, baseline["train_acc"], label="Baseline (CLS) - Train Acc", color="#1f77b4", linestyle="--", linewidth=2)
    plt.plot(epochs, baseline["val_acc"], label="Baseline (CLS) - Val Acc", color="#1f77b4", marker="o", linewidth=2)
    
    plt.plot(epochs, ablation["train_acc"], label="Ablation (GAP) - Train Acc", color="#ff7f0e", linestyle="--", linewidth=2)
    plt.plot(epochs, ablation["val_acc"], label="Ablation (GAP) - Val Acc", color="#ff7f0e", marker="s", linewidth=2)
    
    plt.title("Classification Accuracy Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks(epochs)
    plt.legend(fontsize=11, loc="lower right")
    plt.tight_layout()
    plt.savefig("plots/accuracy_comparison.png")
    plt.close()

if __name__ == "__main__":
    main()
