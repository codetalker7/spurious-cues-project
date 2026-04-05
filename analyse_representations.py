"""
Mechanistic interpretability analysis using saved hidden states and DLA.

Reads the _hs.npz file produced by main2.py alongside the experiment JSON.

Analyses:
    1. Linear probing      — can a linear probe predict the correct answer from
                             hidden states? Compared across baseline / phase1 / phase2b.
    2. Cosine similarity   — do phase1 and phase2b representations move toward
                             the unbiased baseline after self-critique?
    3. Direct Logit Attribution (DLA) — which layers push toward the correct
                             answer vs. the spurious cue (A), and does this
                             shift between phase1 and phase2b?
"""

import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-file', required=True,
                        help="Path to experiment JSON produced by main2.py")
    parser.add_argument('--corrected-only', action='store_true',
                        help="Restrict DLA and similarity plots to corrected examples only")
    return parser.parse_args()


# ── helpers ───────────────────────────────────────────────────────────────────

def mean_cosine_similarity(a, b):
    """Mean cosine similarity between two (n, d) matrices across examples."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return float((a_n * b_n).sum(axis=1).mean())


def probe_layer(X, y, n_splits=5):
    """
    5-fold cross-validated logistic regression accuracy on a single layer.
    Falls back to fewer splits if not enough examples per class.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return np.nan
    n_splits = min(n_splits, int(counts.min()))
    if n_splits < 2:
        return np.nan

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs',
                             multi_class='multinomial')
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
    return float(scores.mean())


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    hs_file = args.experiment_file.replace('.json', '_hs.npz')
    if not os.path.exists(hs_file):
        print(f"Hidden states file not found: {hs_file}")
        print("Run main2.py first — it saves <experiment>_hs.npz alongside the JSON.")
        return

    # ── load data ─────────────────────────────────────────────────────
    with open(args.experiment_file) as f:
        exp = json.load(f)

    hs_data      = np.load(hs_file, allow_pickle=False)
    hs_p1        = hs_data['hs_p1']        # (n, L+1, d)
    hs_p2b       = hs_data['hs_p2b']       # (n, L+1, d)
    hs_baseline  = hs_data['hs_baseline']  # (n, L+1, d)
    dla_p1       = hs_data['dla_p1']       # (n, L, 4)
    dla_p2b      = hs_data['dla_p2b']      # (n, L, 4)
    indices      = hs_data['indices']      # (n,)

    task  = exp['config']['task']
    model = exp['config']['model']
    n, num_layers, d_model = hs_p1.shape
    layers = list(range(num_layers))

    # align JSON records with npz indices
    idx_to_rec = {r['original_index']: r for r in exp['outputs']}
    records = [idx_to_rec[int(i)] for i in indices]

    y_true    = np.array([r['y_true'] for r in records])
    corrected = np.array([
        r['pred_biased'] != r['y_true'] and r['pred_critique'] == r['y_true']
        for r in records
    ], dtype=bool)

    print(f"\nTask:  {task}")
    print(f"Model: {model}")
    print(f"n={n}  corrected={corrected.sum()}  uncorrected={(~corrected).sum()}")

    os.makedirs('figures', exist_ok=True)

    # ── 1. Linear Probing ─────────────────────────────────────────────
    print("\n── Linear Probing ──")
    print("Training a logistic probe at each layer to predict the correct answer.")
    print("High accuracy means the hidden state encodes the correct answer.")

    acc_baseline, acc_p1, acc_p2b = [], [], []

    for layer in range(num_layers):
        acc_baseline.append(probe_layer(hs_baseline[:, layer, :], y_true))
        acc_p1.append(      probe_layer(hs_p1[:,       layer, :], y_true))
        acc_p2b.append(     probe_layer(hs_p2b[:,      layer, :], y_true))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(layers, acc_baseline, label='Baseline (unbiased)',  color='steelblue', linewidth=2)
    ax.plot(layers, acc_p1,       label='Phase 1 (biased)',     color='red',       linewidth=2)
    ax.plot(layers, acc_p2b,      label='Phase 2b (critique)',  color='green',     linewidth=2)
    ax.axhline(1 / len(np.unique(y_true)), color='gray', linestyle='--',
               linewidth=1, label='Chance')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Probe accuracy (CV)')
    ax.set_title(
        f'Linear Probing — does hidden state encode the correct answer?\n'
        f'{task} | {model}\n'
        f'Key: if Phase 2b ≈ Baseline > Phase 1 → self-critique genuinely shifted representation'
    )
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f'figures/probe_{task}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    # print layer-by-layer summary at key layers
    report_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
    print(f"\n  {'Layer':>6} {'Baseline':>10} {'Phase1':>8} {'Phase2b':>9}")
    for l in report_layers:
        print(f"  {l:>6} {acc_baseline[l]:>10.3f} {acc_p1[l]:>8.3f} {acc_p2b[l]:>9.3f}")

    # ── 2. Cosine Similarity to Unbiased Baseline ──────────────────────
    print("\n── Cosine Similarity to Unbiased Baseline ──")
    print("If Phase2b is closer to Baseline than Phase1, self-critique moved")
    print("the representation toward the unbiased state (not just the output).")

    mask = corrected if args.corrected_only else np.ones(n, dtype=bool)

    sim_p1, sim_p2b = [], []
    for layer in range(num_layers):
        sim_p1.append( mean_cosine_similarity(hs_p1[mask,  layer, :],
                                               hs_baseline[mask, layer, :]))
        sim_p2b.append(mean_cosine_similarity(hs_p2b[mask, layer, :],
                                               hs_baseline[mask, layer, :]))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(layers, sim_p1,  label='Phase 1 vs Baseline',  color='red',   linewidth=2)
    ax.plot(layers, sim_p2b, label='Phase 2b vs Baseline', color='green', linewidth=2)
    subset_label = 'corrected examples only' if args.corrected_only else 'all examples'
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean cosine similarity')
    ax.set_title(
        f'Representation Similarity to Unbiased Baseline ({subset_label})\n'
        f'{task} | {model}\n'
        f'Key: Phase2b > Phase1 → self-critique brought reps closer to unbiased state'
    )
    ax.legend()
    ax.set_ylim([0.5, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f'figures/rep_similarity_{task}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    # ── 3. Direct Logit Attribution ───────────────────────────────────
    print("\n── Direct Logit Attribution ──")
    print("Shows which layers push toward each answer option.")
    print("Positive = layer pushes toward that option, negative = pushes away.")

    options  = ['A', 'B', 'C', 'D']
    colors   = ['red', 'steelblue', 'gray', 'gray']

    # split by corrected vs uncorrected for comparison
    dla_layers = list(range(dla_p1.shape[1]))

    for subset_label, mask in [('all', np.ones(n, dtype=bool)),
                                ('corrected', corrected),
                                ('uncorrected', ~corrected)]:
        if mask.sum() < 2:
            continue

        mean_p1  = dla_p1[mask].mean(axis=0)   # (num_layers, 4)
        mean_p2b = dla_p2b[mask].mean(axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        fig.suptitle(
            f'Direct Logit Attribution — {subset_label} examples\n'
            f'{task} | {model}\n'
            f'Key: does Phase 2b reduce (A) contribution and boost correct-answer contribution?'
        )

        for ax, mean_dla, phase_label in zip(axes,
                                              [mean_p1,  mean_p2b],
                                              ['Phase 1 (biased)', 'Phase 2b (critique)']):
            for opt_idx, (opt, col) in enumerate(zip(options, colors)):
                ax.plot(dla_layers, mean_dla[:, opt_idx],
                        label=f'({opt})', color=col,
                        linewidth=2 if opt in ('A',) else 1,
                        alpha=1.0 if opt in ('A',) else 0.5)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Mean logit contribution')
            ax.set_title(phase_label)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2, axis='y')

        plt.tight_layout()
        out = f'figures/dla_{task}_{subset_label}.png'
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out}")

    # ── 4. DLA difference: Phase1 vs Phase2b for corrected examples ────
    if corrected.sum() >= 2:
        diff_dla = dla_p2b[corrected].mean(axis=0) - dla_p1[corrected].mean(axis=0)
        # positive = Phase2b pushes MORE toward that option than Phase1

        fig, ax = plt.subplots(figsize=(11, 5))
        for opt_idx, (opt, col) in enumerate(zip(options, colors)):
            ax.plot(dla_layers, diff_dla[:, opt_idx],
                    label=f'({opt})', color=col,
                    linewidth=2 if opt == 'A' else 1,
                    alpha=1.0 if opt == 'A' else 0.6)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Layer')
        ax.set_ylabel('ΔLogit contribution (Phase2b − Phase1)')
        ax.set_title(
            f'DLA Shift: Phase2b − Phase1 (corrected examples only)\n'
            f'{task} | {model}\n'
            f'Key: negative (A) = Phase2b suppresses spurious cue; '
            f'positive correct = Phase2b boosts correct answer'
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')
        plt.tight_layout()
        out = f'figures/dla_diff_{task}.png'
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  Saved: {out}")

    print("\nDone.")


if __name__ == '__main__':
    main()
