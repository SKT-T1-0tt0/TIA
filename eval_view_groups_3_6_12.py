#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从三组评估报告中筛选并显示指定 tacm 实验 (3, 6, 12) 的结果。

用法:
  python3 eval_view_groups_3_6_12.py
  python3 eval_view_groups_3_6_12.py --input evaluation_report_three_groups.json --output view_3_6_12.txt
  python3 eval_view_groups_3_6_12.py --groups 3 6 12
"""

import os
import sys
import json
import argparse


# 每组配置: (名称, real_dir, baseline_dir, mcfl_dir)
GROUPS = [
    ("post_URMP", "results/1_tacm_/real", "results/3_tacm_/fake1_30fps", "results/5_tacm_/fake1_30fps"),
    ("post_landscape", "results/7_tacm_/real", "results/7_tacm_/fake1_30fps", "results/8_tacm_/fake1_30fps"),
    ("post_audioset_drums", "results/9_tacm_/real", "results/9_tacm_/fake1_30fps", "results/10_tacm_/fake1_30fps"),
]


def group_uses_tacm(name, real_dir, baseline_dir, mcfl_dir, target_nums):
    """判断该组是否涉及任意一个目标 tacm 编号。"""
    paths = [real_dir, baseline_dir, mcfl_dir]
    for p in paths:
        for n in target_nums:
            if f"{n}_tacm_" in p:
                return True
    return False


def format_metric_row(metric_name, metric_results, display_name=None):
    if display_name is None:
        display_name = "FVD-32" if metric_name == "fvd_32" else metric_name.upper()
    row = {"Metric": display_name, "Baseline": "-", "MCFL": "-", "Δ": "-"}
    if "error" in metric_results:
        return row
    if "baseline" in metric_results:
        row["Baseline"] = metric_results["baseline"].get("value", "N/A")
    if "mcfl" in metric_results:
        row["MCFL"] = metric_results["mcfl"].get("value", "N/A")
    if "baseline" in metric_results and "mcfl" in metric_results:
        b, m = metric_results["baseline"], metric_results["mcfl"]
        bmean, mmean = b.get("mean"), m.get("mean")
        if bmean is not None and mmean is not None:
            row["Δ"] = f"{mmean - bmean:+.4f}"
    return row


def format_tc_flicker_rows(metric_results):
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


def write_filtered_report(report, filtered_names, output_file):
    results = report.get("results", {})
    metrics = ["fvd", "fvd_32", "fid", "ffc", "clip", "av_align", "tc_flicker"]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"评估结果筛选: tacm 实验 3, 6, 12\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {report.get('timestamp', 'N/A')}\n")
        f.write("\n")

        for name, real_dir, baseline_dir, mcfl_dir in GROUPS:
            if name not in filtered_names:
                continue
            f.write(f"【{name}】\n")
            f.write(f"  real:     {real_dir}\n")
            f.write(f"  baseline: {baseline_dir}\n")
            f.write(f"  mcfl:     {mcfl_dir}\n")
        f.write("=" * 80 + "\n\n")

        for group_name in filtered_names:
            if group_name not in results:
                continue
            group_results = results[group_name]
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write(f"  {group_name}\n")
            f.write("=" * 80 + "\n\n")

            table_data = []
            for metric_name in metrics:
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


def main():
    parser = argparse.ArgumentParser(description="从三组评估报告中筛选 3, 6, 12 的数据结果")
    parser.add_argument("--input", type=str, default="evaluation_report_three_groups.json",
                        help="输入的 JSON 报告路径")
    parser.add_argument("--output", type=str, default="evaluation_report_3_6_12.txt",
                        help="筛选后的 txt 报告路径")
    parser.add_argument("--groups", type=int, nargs="+", default=[3, 6, 12],
                        help="要筛选的 tacm 编号 (默认: 3 6 12)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        print("   请先运行: python3 eval_all_three_groups.py --metrics fvd fid ffc clip av_align tc_flicker --output evaluation_report_three_groups.txt")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        report = json.load(f)

    filtered_names = []
    for name, real_dir, baseline_dir, mcfl_dir in GROUPS:
        if group_uses_tacm(name, real_dir, baseline_dir, mcfl_dir, args.groups):
            filtered_names.append(name)

    if not filtered_names:
        print(f"⚠️ 没有找到涉及 tacm {args.groups} 的组。")
        print("   当前配置中:")
        for name, real_dir, baseline_dir, mcfl_dir in GROUPS:
            print(f"   - {name}: real={real_dir}, baseline={baseline_dir}, mcfl={mcfl_dir}")
        sys.exit(1)

    print(f"📋 筛选 tacm {args.groups} 涉及的组: {filtered_names}")
    write_filtered_report(report, filtered_names, args.output)
    print(f"💾 已保存: {args.output}")


if __name__ == "__main__":
    main()
