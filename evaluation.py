import json
import os
from glob import glob

# Folder containing your JSON files
folder_path = "./qwen3-1.7B_thinking_generations"  # change this to your folder path

# Find all JSON files
json_files = glob(os.path.join(folder_path, "*.json"))

# Initialize overall counters
overall_metrics = {
    "baseline_correct": 0,
    "biased_correct": 0,
    "review_correct": 0,
    "critique_correct": 0,
    "total": 0,
    "corrections": 0
}

print("Per-file accuracy metrics:\n")

for fname in json_files:
    with open(fname, "r") as f:
        data = json.load(f)

    outputs = data["outputs"]
    metrics = {
        "baseline_correct": 0,
        "biased_correct": 0,
        "review_correct": 0,
        "critique_correct": 0,
        "total": len(outputs),
        "corrections": 0
    }

    for out in outputs:
        y_true = out["y_true"]

        if out["pred_baseline"] == y_true:
            metrics["baseline_correct"] += 1
        if out["pred_biased"] == y_true:
            metrics["biased_correct"] += 1
        if out["pred_review"] == y_true:
            metrics["review_correct"] += 1
        if out["pred_critique"] == y_true:
            metrics["critique_correct"] += 1

        if (out["pred_baseline"] != y_true) and (out["pred_critique"] == y_true):
            metrics["corrections"] += 1

    # Print per-file accuracy
    baseline_acc = metrics["baseline_correct"] / metrics["total"]
    biased_acc = metrics["biased_correct"] / metrics["total"]
    review_acc = metrics["review_correct"] / metrics["total"]
    critique_acc = metrics["critique_correct"] / metrics["total"]

    print(f"File: {os.path.basename(fname)}")
    print(f"  baseline_acc: {baseline_acc:.4f}")
    print(f"  biased_acc: {biased_acc:.4f}")
    print(f"  review_acc: {review_acc:.4f}")
    print(f"  critique_acc: {critique_acc:.4f}")
    print(f"  corrections by critique: {metrics['corrections']}\n")

    # Update overall totals
    overall_metrics["baseline_correct"] += metrics["baseline_correct"]
    overall_metrics["biased_correct"] += metrics["biased_correct"]
    overall_metrics["review_correct"] += metrics["review_correct"]
    overall_metrics["critique_correct"] += metrics["critique_correct"]
    overall_metrics["corrections"] += metrics["corrections"]
    overall_metrics["total"] += metrics["total"]

# Compute overall accuracies
overall_baseline_acc = overall_metrics["baseline_correct"] / overall_metrics["total"]
overall_biased_acc = overall_metrics["biased_correct"] / overall_metrics["total"]
overall_review_acc = overall_metrics["review_correct"] / overall_metrics["total"]
overall_critique_acc = overall_metrics["critique_correct"] / overall_metrics["total"]

print("=== Overall Accuracy Metrics ===")
print(f"Overall baseline_acc: {overall_baseline_acc:.4f}")
print(f"Overall biased_acc: {overall_biased_acc:.4f}")
print(f"Overall review_acc: {overall_review_acc:.4f}")
print(f"Overall critique_acc: {overall_critique_acc:.4f}")
print(f"Total corrections by critique: {overall_metrics['corrections']}")
