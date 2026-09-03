#!/usr/bin/env python3
"""Plot the pure VV attention sparsity upper bound from M1 oracle traces."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


KEEP_RATIOS = (0.10, 0.20, 0.25, 0.35, 0.50, 0.75)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "query_kind",
        "num_video_keys",
        "dit_index",
        "layer_index",
        "top_p_token_count_mean_p90",
        "top_p_token_count_p95_p90",
    ]
    for ratio in KEEP_RATIOS:
        suffix = f"{int(round(100 * ratio)):03d}"
        columns.extend((f"mass_mean_r{suffix}", f"mass_p05_r{suffix}"))

    table = pq.read_table(
        args.input,
        columns=columns,
        filters=[("query_kind", "=", "video")],
    )
    frame = table.to_pandas()
    nkeys = frame["num_video_keys"].astype(float)
    frame["keep_mean_q_m90"] = frame["top_p_token_count_mean_p90"] / nkeys
    frame["keep_p95_q_m90"] = frame["top_p_token_count_p95_p90"] / nkeys

    layer_rows = []
    for layer, group in frame.groupby("layer_index", sort=True):
        robust = group["keep_p95_q_m90"]
        mean_q = group["keep_mean_q_m90"]
        mass20_p05 = group["mass_p05_r020"]
        layer_rows.append(
            {
                "layer": int(layer),
                "records": len(group),
                "keep_mean_query_m90_mean": mean_q.mean(),
                "sparsity_mean_query_m90_mean": 1.0 - mean_q.mean(),
                "keep_p95_query_m90_mean": robust.mean(),
                "keep_p95_query_m90_p25": robust.quantile(0.25),
                "keep_p95_query_m90_p50": robust.quantile(0.50),
                "keep_p95_query_m90_p75": robust.quantile(0.75),
                "keep_p95_query_m90_p90": robust.quantile(0.90),
                "sparsity_p95_query_m90_mean": 1.0 - robust.mean(),
                "sparsity_p95_query_m90_p25": 1.0 - robust.quantile(0.75),
                "sparsity_p95_query_m90_p75": 1.0 - robust.quantile(0.25),
                "mass20_mean": group["mass_mean_r020"].mean(),
                "mass20_p05_query_mean": mass20_p05.mean(),
                "mass20_p05_query_pass_rate": (mass20_p05 >= 0.90).mean(),
            }
        )
    layers = pd.DataFrame(layer_rows)
    layers.to_csv(args.output_dir / "vv_layer_sparsity.csv", index=False)

    budget_rows = []
    for ratio in KEEP_RATIOS:
        suffix = f"{int(round(100 * ratio)):03d}"
        mass_mean = frame[f"mass_mean_r{suffix}"]
        mass_p05 = frame[f"mass_p05_r{suffix}"]
        budget_rows.append(
            {
                "keep_ratio": ratio,
                "sparsity": 1.0 - ratio,
                "mass_mean": mass_mean.mean(),
                "mass_p05_query_mean": mass_p05.mean(),
                "fraction_records_mass_p05_ge_090": (mass_p05 >= 0.90).mean(),
            }
        )
    budgets = pd.DataFrame(budget_rows)
    budgets.to_csv(args.output_dir / "vv_fixed_budget_summary.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    x = layers["layer"].to_numpy()
    ax = axes[0]
    ax.plot(
        x,
        100 * layers["sparsity_mean_query_m90_mean"],
        color="#4C78A8",
        linewidth=2.0,
        label="Mean query oracle",
    )
    ax.plot(
        x,
        100 * layers["sparsity_p95_query_m90_mean"],
        color="#E45756",
        linewidth=2.5,
        label="95%-query coverage oracle",
    )
    ax.fill_between(
        x,
        100 * layers["sparsity_p95_query_m90_p25"],
        100 * layers["sparsity_p95_query_m90_p75"],
        color="#E45756",
        alpha=0.16,
        label="Interquartile range across head states",
    )
    ax.axhline(80.0, color="#222222", linestyle="--", linewidth=1.4, label="80% sparsity target")
    ax.axvspan(-0.5, 11.5, color="#F2CF5B", alpha=0.08)
    ax.axvspan(11.5, 27.5, color="#54A24B", alpha=0.06)
    ax.axvspan(27.5, 39.5, color="#B279A2", alpha=0.07)
    ax.set_ylabel("Oracle key sparsity (%)")
    ax.set_ylim(0, 100)
    ax.set_title("DreamZero VV attention: layer-wise sparsity upper bound (90% retained mass)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", ncol=2, frameon=False)

    ax = axes[1]
    ax.plot(x, layers["mass20_mean"], color="#4C78A8", linewidth=2.0, label="Mean query mass")
    ax.plot(
        x,
        layers["mass20_p05_query_mean"],
        color="#E45756",
        linewidth=2.5,
        label="Mean of per-head 5th-percentile query mass",
    )
    ax.axhline(0.90, color="#222222", linestyle="--", linewidth=1.4, label="Quality gate = 0.90")
    ax.set_ylabel("Retained attention mass at 20% keys")
    ax.set_xlabel("DiT layer index")
    ax.set_ylim(0.45, 1.01)
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(
        x,
        100 * layers["mass20_p05_query_pass_rate"],
        color="#54A24B",
        linewidth=1.8,
        linestyle=":",
        label="Head-state pass rate",
    )
    ax2.set_ylabel("Head-state pass rate (%)", color="#397B34")
    ax2.set_ylim(0, 100)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "vv_layer_sparsity.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output_dir / "vv_layer_sparsity.pdf", bbox_inches="tight")
    plt.close(fig)

    def subset_summary(name: str, mask: pd.Series) -> dict[str, object]:
        group = frame.loc[mask]
        robust = group["keep_p95_q_m90"]
        mass20 = group["mass_p05_r020"]
        return {
            "stage": name,
            "keep": robust.mean(),
            "sparsity": 1.0 - robust.mean(),
            "mass20": mass20.mean(),
            "pass20": (mass20 >= 0.90).mean(),
        }

    stage_rows = [
        subset_summary("Early (0-11)", frame["layer_index"].between(0, 11)),
        subset_summary("Middle (12-27)", frame["layer_index"].between(12, 27)),
        subset_summary("Late (28-39)", frame["layer_index"].between(28, 39)),
    ]
    overall_mean_q = frame["keep_mean_q_m90"].mean()
    overall_robust = frame["keep_p95_q_m90"].mean()
    robust_median = frame["keep_p95_q_m90"].median()
    mass20_mean = frame["mass_mean_r020"].mean()
    mass20_p05 = frame["mass_p05_r020"].mean()
    pass20 = (frame["mass_p05_r020"] >= 0.90).mean()

    report = [
        "# VV Attention Sparsity Audit",
        "",
        "This audit isolates `video-query -> video-key` self-attention. For each sampled query,",
        "`m_q(S) = sum_{k in S} softmax(Q_v K_v^T / sqrt(d))_qk`.",
        "The top-p oracle chooses the best keys independently for each query, so it is an optimistic",
        "upper bound rather than the cost of a deployable shared mask.",
        "",
        "## Dataset",
        "",
        f"- VV head-state records: {len(frame):,}",
        f"- Video keys per record: {int(frame['num_video_keys'].iloc[0]):,}",
        "- Query sampling: 32 video queries per head state",
        "- Coverage: 8 DiT steps, 40 layers, 40 heads, both CFG branches",
        "",
        "## Main results",
        "",
        "| Measurement | Required keys | Implied key sparsity |",
        "|---|---:|---:|",
        f"| Mean query, retain 90% mass | {pct(overall_mean_q)} | {pct(1-overall_mean_q)} |",
        f"| Cover 95% of queries in a head state, retain 90% mass (mean) | {pct(overall_robust)} | {pct(1-overall_robust)} |",
        f"| Same robust measurement, median head state | {pct(robust_median)} | {pct(1-robust_median)} |",
        "",
        "At a fixed 20% key budget:",
        "",
        "| Mean retained mass | Mean 5th-percentile query mass | Head-state pass rate (`p05 >= 0.90`) |",
        "|---:|---:|---:|",
        f"| {mass20_mean:.4f} | {mass20_p05:.4f} | {pct(pass20)} |",
        "",
        "## Layer-stage summary",
        "",
        "| Layer stage | Keys for 95%-query coverage | Implied sparsity | p05 query mass at 20% keys | Pass rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in stage_rows:
        report.append(
            f"| {row['stage']} | {pct(float(row['keep']))} | {pct(float(row['sparsity']))} | "
            f"{float(row['mass20']):.4f} | {pct(float(row['pass20']))} |"
        )
    report.extend(
        [
            "",
            "## Fixed-budget table",
            "",
            "| Keep ratio | Sparsity | Mean retained mass | Mean p05 query mass | Pass rate |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in budget_rows:
        report.append(
            f"| {pct(row['keep_ratio'])} | {pct(row['sparsity'])} | {row['mass_mean']:.4f} | "
            f"{row['mass_p05_query_mean']:.4f} | {pct(row['fraction_records_mass_p05_ge_090'])} |"
        )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "The VV matrix is meaningfully sparse, but a uniform 80% sparsity policy is unsafe:",
            "roughly half of the head states fail the worst-query-tail mass gate. Middle layers are",
            "the strongest sparse region, while late layers are substantially more diffuse. M1 should",
            "therefore predict a continuous budget by step/layer/head and retain dense fallback.",
            "",
            "![Layer-wise VV sparsity](vv_layer_sparsity.png)",
            "",
            "## Reproducibility",
            "",
            "- `vv_layer_sparsity.csv`: complete 40-layer table.",
            "- `vv_fixed_budget_summary.csv`: fixed-budget quality table.",
            "- `vv_layer_sparsity.pdf`: vector figure for the paper.",
        ]
    )
    (args.output_dir / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
