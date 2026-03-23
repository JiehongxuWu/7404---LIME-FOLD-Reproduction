#!/usr/bin/env python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    datasets = ["heart", "wine", "voting"]
    rows = []
    for ds in datasets:
        p = LOGS_DIR / f"{ds}_lime_fold_results.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if df.empty:
            continue
        r = df.iloc[0]
        rows.append(
            {
                "Data Set": ds,
                "Prec.": f"{float(r['precision']):.3f}",
                "Recall": f"{float(r['recall']):.3f}",
                "Acc.": f"{float(r['accuracy']):.3f}",
                "F1": f"{float(r['f1']):.3f}",
                "Avg Rules": f"{float(r['avg_num_rules']):.2f}",
            }
        )

    if not rows:
        print("No dataset result files found.")
        return

    out_df = pd.DataFrame(rows)
    out_csv = LOGS_DIR / "table1_ours.csv"
    out_df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, max(2.5, 1.2 + 0.8 * len(out_df))))
    ax.axis("off")
    table = ax.table(
        cellText=out_df.values,
        colLabels=out_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.1, 1.5)
    ax.set_title("Table 1 Style - Our Results (LIME-FOLD, 5-fold CV)", fontsize=13, pad=12)
    fig.tight_layout()
    out_png = FIGURES_DIR / "table1_ours.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()

