import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-file', required=True,
                        help="Path to experiment JSON file produced by main.py")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.experiment_file) as f:
        data = json.load(f)

    outputs  = data['outputs']
    task     = data['config']['task']
    model    = data['config']['model']

    print(f"\nTask:  {task}")
    print(f"Model: {model}")
    print(f"Total examples: {len(outputs)}")

    # ── collect flip layer differences ──────────────────────────
    flip_diffs = []

    for rec in outputs:
        f1  = rec.get('flip_layer_p1')
        f2b = rec.get('flip_layer_p2b')

        # skip examples where logit lens was not run
        if f1 is None and f2b is None:
            continue

        corrected = (
            rec['pred_biased']   != rec['y_true'] and
            rec['pred_critique'] == rec['y_true']
        )

        # if model never predicted correct answer in phase 1
        # treat flip_layer as last layer (worst case)
        num_layers = len(rec.get('logit_lens_p1', []))
        f1_safe  = f1  if f1  is not None else num_layers
        f2b_safe = f2b if f2b is not None else num_layers

        # positive diff = Phase 2b flips to correct answer earlier = good
        diff = f1_safe - f2b_safe

        flip_diffs.append({
            'diff':      diff,
            'corrected': corrected,
            'flip_p1':   f1_safe,
            'flip_p2b':  f2b_safe,
        })

    if not flip_diffs:
        print("\nNo logit lens data found.")
        print("Run main.py with --logit-lens flag first.")
        return

    # ── split into corrected vs uncorrected ─────────────────────
    corrected_diffs   = [d['diff'] for d in flip_diffs if     d['corrected']]
    uncorrected_diffs = [d['diff'] for d in flip_diffs if not d['corrected']]

    corrected_p1    = [d['flip_p1']  for d in flip_diffs if     d['corrected']]
    corrected_p2b   = [d['flip_p2b'] for d in flip_diffs if     d['corrected']]
    uncorrected_p1  = [d['flip_p1']  for d in flip_diffs if not d['corrected']]
    uncorrected_p2b = [d['flip_p2b'] for d in flip_diffs if not d['corrected']]

    # ── print summary table ──────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"{'Subset':<20} {'n':>4} {'Avg flip P1':>12} {'Avg flip P2b':>13} {'Avg diff':>10}")
    print(f"{'─'*50}")

    if corrected_diffs:
        print(f"{'Corrected':<20} "
              f"{len(corrected_diffs):>4} "
              f"{np.mean(corrected_p1):>12.1f} "
              f"{np.mean(corrected_p2b):>13.1f} "
              f"{np.mean(corrected_diffs):>10.1f}")

    if uncorrected_diffs:
        print(f"{'Uncorrected':<20} "
              f"{len(uncorrected_diffs):>4} "
              f"{np.mean(uncorrected_p1):>12.1f} "
              f"{np.mean(uncorrected_p2b):>13.1f} "
              f"{np.mean(uncorrected_diffs):>10.1f}")

    print(f"{'─'*50}")
    print("\nHow to read this:")
    print("  Avg flip P1  = which layer Phase 1 first predicts correct answer")
    print("  Avg flip P2b = which layer Phase 2b first predicts correct answer")
    print("  Avg diff     = positive means Phase 2b flips earlier (good)")
    print("  Key result   = corrected diff >> uncorrected diff → mechanistic evidence")

    # ── plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Logit Lens Summary | {task} | {model}", fontsize=12)

    # Plot 1: distribution of flip layer differences
    ax = axes[0]
    if corrected_diffs:
        ax.hist(corrected_diffs,   bins=10, alpha=0.7,
                color='green', label=f'Corrected (n={len(corrected_diffs)})')
    if uncorrected_diffs:
        ax.hist(uncorrected_diffs, bins=10, alpha=0.7,
                color='red',   label=f'Uncorrected (n={len(uncorrected_diffs)})')
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Flip layer difference (P1 − P2b)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of flip layer differences\n'
                 'positive = P2b flips to correct answer earlier')
    ax.legend()
    ax.grid(True, alpha=0.2)

    # Plot 2: avg flip layer P1 vs P2b side by side bars
    ax = axes[1]
    groups = []
    vals_p1, vals_p2b, labels = [], [], []

    if corrected_diffs:
        vals_p1.append(np.mean(corrected_p1))
        vals_p2b.append(np.mean(corrected_p2b))
        labels.append('Corrected')
    if uncorrected_diffs:
        vals_p1.append(np.mean(uncorrected_p1))
        vals_p2b.append(np.mean(uncorrected_p2b))
        labels.append('Uncorrected')

    x     = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, vals_p1,  width, label='Phase 1',  color='lightblue', edgecolor='steelblue')
    ax.bar(x + width/2, vals_p2b, width, label='Phase 2b', color='lightsalmon', edgecolor='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Avg layer where correct answer first appears')
    ax.set_title('When does correct answer emerge?\n'
                 'lower P2b bar = earlier emergence = genuine correction')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out_path = f'figures/logit_lens_summary_{task}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {out_path}")

if __name__ == '__main__':
    main()
