#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三组数据集对比评估脚本
- post_URMP:       real=3_tacm_, baseline=3_tacm_, mcfl=6_tacm_
- post_landscape:  real=7_tacm_, baseline=7_tacm_, mcfl=8_tacm_
- post_audioset_drums: real=9_tacm_, baseline=9_tacm_, mcfl=10_tacm_

三组结果合并输出到同一报告。

用法:
  python3 eval_all_three_groups.py --metrics fvd fid ffc clip av_align tc_flicker --output evaluation_report_three_groups.txt
"""

import os
import sys
import json
from datetime import datetime

# 确保当前目录在 path 中，以便 import eval_all
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_all import EvaluationRunner


# 三组配置: (名称, real_dir, baseline_dir, mcfl_dir)
GROUPS = [
    (
        "post_URMP",
        "results/3_tacm_/real",
        "results/3_tacm_/fake1_30fps",
        "results/6_tacm_/fake1_30fps",
    ),
    (
        "post_landscape",
        "results/7_tacm_/real",
        "results/7_tacm_/fake1_30fps",
        "results/8_tacm_/fake1_30fps",
    ),
    (
        "post_audioset_drums",
        "results/9_tacm_/real",
        "results/9_tacm_/fake1_30fps",
        "results/10_tacm_/fake1_30fps",
    ),
]


def format_metric_row(metric_name, metric_results, display_name=None):
    """从单组 results 中格式化为 Baseline / MCFL / Δ 一行."""
    if display_name is None:
        display_name = "FVD-32" if metric_name == "fvd_32" else metric_name.upper()
    row = {"Metric": display_name, "Baseline": "-", "MCFL": "-", "Δ": "-"}
    if "error" in metric_results:
        return row
    if "baseline" in metric_results:
        r = metric_results["baseline"]
        row["Baseline"] = r.get("value", "N/A")
    if "mcfl" in metric_results:
        r = metric_results["mcfl"]
        row["MCFL"] = r.get("value", "N/A")
    if "baseline" in metric_results and "mcfl" in metric_results:
        b, m = metric_results["baseline"], metric_results["mcfl"]
        bmean = b.get("mean")
        mmean = m.get("mean")
        if bmean is not None and mmean is not None:
            row["Δ"] = f"{mmean - bmean:+.4f}"
    return row


def format_tc_flicker_rows(metric_results):
    """TC-Flicker: 输出 mean 与 median 两行（小样本时 median 更稳健）。"""
    rows = []
    for sub in ["flicker", "tc"]:
        for agg in ["mean", "median"]:
            label = "median" if agg == "median" else "mean"
            r = {
                "Metric": f"TC_FLICKER ({sub}, {label})",
                "Baseline": "-",
                "MCFL": "-",
                "Δ": "-",
            }
            if "error" in metric_results:
                rows.append(r)
                continue
            b = metric_results.get("baseline", {}).get(sub, {})
            m = metric_results.get("mcfl", {}).get(sub, {})
            if b:
                r["Baseline"] = b.get("value_median" if agg == "median" else "value", b.get("value", "N/A"))
            if m:
                r["MCFL"] = m.get("value_median" if agg == "median" else "value", m.get("value", "N/A"))
            if b and m and agg in b and agg in m:
                r["Δ"] = f"{m[agg] - b[agg]:+.4f}"
            rows.append(r)
    return rows


def write_combined_report(all_group_results, output_file, metrics):
    """将三组结果写入一个 txt 和一个 json."""
    timestamp = datetime.now().isoformat()
    report = {
        "timestamp": timestamp,
        "groups": list(g[0] for g in GROUPS),
        "results": all_group_results,
    }

    # JSON
    json_file = output_file.replace(".txt", ".json") if output_file.endswith(".txt") else output_file + ".json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # TXT
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("三组数据集对比评估报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {timestamp}\n")
        f.write("\n")
        for (name, real_dir, baseline_dir, mcfl_dir) in GROUPS:
            f.write(f"【{name}】\n")
            f.write(f"  real:     {real_dir}\n")
            f.write(f"  baseline: {baseline_dir}\n")
            f.write(f"  mcfl:     {mcfl_dir}\n")
        f.write("=" * 80 + "\n\n")

        for group_name, group_results in all_group_results.items():
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(f"  {group_name}\n")
            f.write("=" * 80 + "\n\n")

            table_data = []
            for metric_name in metrics:
                metric_name = metric_name.lower()
                if metric_name not in group_results:
                    continue
                metric_results = group_results[metric_name]
                if metric_name == "tc_flicker":
                    table_data.extend(format_tc_flicker_rows(metric_results))
                else:
                    display_name = "FVD-32" if metric_name == "fvd_32" else metric_name.upper()
                    table_data.append(format_metric_row(metric_name, metric_results, display_name))

            if table_data:
                wm = max(len(r["Metric"]) for r in table_data) + 2
                wb = max(len(r["Baseline"]) for r in table_data) + 2
                wc = max(len(r["MCFL"]) for r in table_data) + 2
                wd = max(len(r["Δ"]) for r in table_data) + 2
                header = f"{'Metric':<{wm}} | {'Baseline':<{wb}} | {'MCFL':<{wc}} | {'Δ':<{wd}}"
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                for row in table_data:
                    f.write(f"{row['Metric']:<{wm}} | {row['Baseline']:<{wb}} | {row['MCFL']:<{wc}} | {row['Δ']:<{wd}}\n")
            f.write("\n")

        # 小样本说明（任一组 TC_FLICKER 视频数 ≤20 时）
        n_note = None
        for group_results in all_group_results.values():
            mr = group_results.get("tc_flicker", {})
            if "error" in mr:
                continue
            for key in ("baseline", "mcfl"):
                if mr.get(key, {}).get("n_videos") is not None and mr[key]["n_videos"] <= 20:
                    n_note = mr[key]["n_videos"]
                    break
            if n_note is not None:
                break
        if n_note is not None:
            f.write("=" * 80 + "\n")
            f.write("[小样本说明] 当前部分组 TC_FLICKER 评估视频数 N=%d ≤ 20。此时 mean 易受单条异常值影响，不能据此断定 baseline 比 MCFL 更稳定；请优先参考 median。若 median 上 MCFL 与 baseline 接近或更优，说明多数视频时间一致性相当或更好。\n" % n_note)
            f.write("=" * 80 + "\n")

    print(f"\n💾 合并报告已保存: {output_file}")
    print(f"💾 JSON 已保存: {json_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="三组数据集对比评估 (post_URMP / post_landscape / post_audioset_drums)")
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=["fvd", "fid", "ffc", "clip", "av_align", "tc_flicker"],
                        choices=["fvd", "fvd_32", "fid", "ffc", "clip", "av_align", "tc_flicker"],
                        help="要运行的评估指标")
    parser.add_argument("--output", type=str, default="evaluation_report_three_groups.txt",
                        help="合并报告输出路径")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_bootstrap", action="store_true")
    parser.add_argument("--bootstrap_k", type=int, default=10)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--prompt_file", type=str, default="prompts.txt")
    args = parser.parse_args()

    metrics = [m.lower() for m in args.metrics]
    all_group_results = {}

    for name, real_dir, baseline_dir, mcfl_dir in GROUPS:
        if not os.path.exists(real_dir):
            print(f"⚠️ 跳过 {name}: 真实目录不存在 {real_dir}")
            continue
        if not os.path.exists(baseline_dir):
            print(f"⚠️ 跳过 {name}: Baseline 目录不存在 {baseline_dir}")
            continue
        if not os.path.exists(mcfl_dir):
            print(f"⚠️ 跳过 {name}: MCFL 目录不存在 {mcfl_dir}")
            continue

        print("\n" + "=" * 80)
        print(f"📂 正在评估: {name}")
        print("=" * 80)

        runner = EvaluationRunner(
            real_dir=real_dir,
            baseline_dir=baseline_dir,
            mcfl_dir=mcfl_dir,
            prompt_file=args.prompt_file,
            device=args.device,
            use_bootstrap=not args.no_bootstrap,
            bootstrap_k=args.bootstrap_k,
            use_cache=not args.no_cache,
            output_file=None,
        )
        runner.run_all(metrics)
        runner.print_report()
        all_group_results[name] = runner.results

    if not all_group_results:
        print("❌ 没有成功完成任何一组的评估。")
        return

    write_combined_report(all_group_results, args.output, metrics)


if __name__ == "__main__":
    main()
