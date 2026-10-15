# -*- coding: utf-8 -*-
"""
为本科毕业论文《重排序模块在RAG系统中对生成质量的增效研究与实证分析》生成配图。

生成图片：
1. fig1_rag_pipeline.png              本文RAG系统整体实验流程
2. fig2_rerank_strategy_compare.png   不同重排序策略的工作机制对比
3. fig3_dataset_task_structure.png    三类实验数据集的任务结构差异
4. fig4_f1_comparison.png             不同重排序策略在三个数据集上的F1表现对比
5. fig5_multihoprag_metric_gap.png    MultiHop-RAG中检索指标与生成指标的差异

运行方式：
python make_thesis_figures.py

依赖：
pip install matplotlib numpy
"""

import argparse
from pathlib import Path
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "thesis_figures"


def setup_chinese_font():
    """
    尽量自动匹配常见中文字体。
    如果运行后中文乱码，请手动把 font_candidates 中的字体名改成你电脑已有的中文字体。
    Windows 常见：SimHei, Microsoft YaHei
    macOS 常见：PingFang SC, Heiti SC
    Linux 常见：Noto Sans CJK SC, WenQuanYi Micro Hei
    """
    bundled_font = PROJECT_ROOT / "wqy-microhei.ttc"
    if bundled_font.is_file():
        font_manager.fontManager.addfont(str(bundled_font))
    plt.rcParams["font.sans-serif"] = [
        "WenQuanYi Micro Hei"
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160


def add_box(ax, x, y, w, h, text, fontsize=11, linewidth=1.5):
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=linewidth,
        facecolor="white",
        edgecolor="black",
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def add_arrow(ax, start, end, linewidth=1.4):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", linewidth=linewidth, color="black"),
    )


def save_fig(fig, filename):
    fig.savefig(OUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def fig1_rag_pipeline():
    """图1 本文RAG系统整体实验流程"""
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    ax.text(5, 11.55, "图1 本文RAG系统整体实验流程", ha="center", va="center", fontsize=16, fontweight="bold")

    boxes = [
        (3.1, 10.2, 3.8, 0.75, "原始数据集\nMultifieldQA / Dureader / MultiHop-RAG"),
        (3.1, 8.9, 3.8, 0.75, "文本清洗与分块\nChunk Size / Overlap"),
        (3.1, 7.6, 3.8, 0.75, "BGE-M3文本向量化\n生成稠密向量表示"),
        (3.1, 6.3, 3.8, 0.75, "ChromaDB向量数据库\n候选文本块召回"),
        (3.1, 4.65, 3.8, 1.05, "重排序模块\nColBERT / Pointwise / Pairwise\nListwise / SetSelection"),
        (3.1, 3.1, 3.8, 0.75, "Top-k证据文本块\n输入生成模型"),
        (3.1, 1.8, 3.8, 0.75, "Deepseek生成答案\nGrounded Answer Generation"),
        (3.1, 0.5, 3.8, 0.75, "系统评估\nF1 / EM / Hrate@1 / MRR / LLM-as-Judge"),
    ]

    for x, y, w, h, text in boxes:
        add_box(ax, x, y, w, h, text, fontsize=11)

    for i in range(len(boxes) - 1):
        x1, y1, w1, h1, _ = boxes[i]
        x2, y2, w2, h2, _ = boxes[i + 1]
        add_arrow(ax, (x1 + w1 / 2, y1), (x2 + w2 / 2, y2 + h2))

    save_fig(fig, "fig1_rag_pipeline.png")


def fig2_rerank_strategy_compare():
    """图2 不同重排序策略的工作机制对比"""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6, 7.55, "图2 不同重排序策略的工作机制对比", ha="center", va="center", fontsize=16, fontweight="bold")

    strategies = [
        ("ColBERT\n后期交互式", "Query token\n↔\nPassage token\nMaxSim打分"),
        ("Pointwise\n点对式", "Query + Passage\n→\n相关性分数"),
        ("Pairwise\n配对式", "Query + A + B\n→\n判断A/B谁更相关"),
        ("Listwise\n序列式", "Query + 多个Passage\n→\n输出整体排序"),
        ("SetSelection\n集合选择", "Query + 多个Passage\n→\n选择互补证据集合"),
    ]

    x_positions = [0.6, 2.9, 5.2, 7.5, 9.8]
    for x, (title, desc) in zip(x_positions, strategies):
        add_box(ax, x, 4.7, 1.8, 1.0, title, fontsize=10.5)
        add_arrow(ax, (x + 0.9, 4.7), (x + 0.9, 3.9))
        add_box(ax, x, 2.2, 1.8, 1.55, desc, fontsize=9.5)

    ax.text(
        6,
        1.25,
        "核心差异：从局部匹配、逐块打分、成对比较，逐步发展到全局排序与证据集合优化。",
        ha="center",
        va="center",
        fontsize=12,
    )

    save_fig(fig, "fig2_rerank_strategy_compare.png")


def fig3_dataset_task_structure():
    """图3 三类实验数据集的任务结构差异"""
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(6, 7.55, "图3 三类实验数据集的任务结构差异", ha="center", va="center", fontsize=16, fontweight="bold")

    rows = [
        ("MultifieldQA\n单跳问答", ["一个问题", "一个主要证据", "一个答案"]),
        ("Dureader\n多跳开放问答", ["一个问题", "多个离散证据", "综合答案"]),
        ("MultiHop-RAG\n多跳证据检索", ["一个问题", "多个候选证据", "证据链组合", "答案"]),
    ]

    y_positions = [5.7, 3.7, 1.7]
    for y, (dataset, steps) in zip(y_positions, rows):
        add_box(ax, 0.45, y, 2.2, 0.8, dataset, fontsize=10.5)
        last_x = 3.1
        for idx, step in enumerate(steps):
            w = 1.65 if idx != 2 else 1.85
            add_box(ax, last_x, y, w, 0.8, step, fontsize=10.5)
            if idx < len(steps) - 1:
                add_arrow(ax, (last_x + w, y + 0.4), (last_x + w + 0.45, y + 0.4))
            last_x += w + 0.7

    save_fig(fig, "fig3_dataset_task_structure.png")


def fig4_f1_comparison():
    """图4 不同重排序策略在三个数据集上的F1表现对比"""
    setup_chinese_font()

    datasets = {
        "MultifieldQA": {
            "Baseline": 44.58,
            "ColBERT": 53.33,
            "Pointwise": 53.44,
            "Pairwise": 59.29,
            "Listwise": 63.86,
            "Contextual+\nPointwise": 60.41,
            "ColBERT+\nListwise": 57.86,
            "Contextual+\nListwise": 65.81,
        },
        "Dureader": {
            "Baseline": 30.08,
            "Pointwise": 29.56,
            "Decompose+\nPointwise": 30.53,
            "ColBERT": 31.85,
            "Listwise": 32.74,
            "SetSelection": 32.96,
        },
        "MultiHop-RAG": {
            "Baseline": 32.27,
            "ColBERT": 34.76,
            "Pointwise": 34.54,
            "Listwise": 44.73,
            "SetSelection": 41.01,
        },
    }

    for dataset, values in datasets.items():
        fig, ax = plt.subplots(figsize=(11, 5.8))
        methods = list(values.keys())
        scores = list(values.values())
        x = np.arange(len(methods))

        bars = ax.bar(x, scores)
        ax.set_title(f"图4 不同重排序策略在{dataset}数据集上的F1表现对比", fontsize=15, fontweight="bold")
        ax.set_ylabel("F1")
        ax.set_ylim(0, max(scores) * 1.25)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{score:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
        fig.tight_layout()
        save_fig(fig, f"fig4_f1_comparison_{dataset.lower().replace('-', '_')}.png")


def fig5_multihoprag_metric_gap():
    """图5 MultiHop-RAG中检索指标与生成指标的差异"""
    methods = ["Baseline", "ColBERT", "Pointwise", "Listwise", "SetSelection"]
    hrate = [30.00, 34.00, 52.00, 52.00, 56.00]
    mrr = [44.33, 48.33, 64.67, 64.50, 70.17]
    f1 = [32.27, 34.76, 34.54, 44.73, 41.01]
    em = [32.00, 34.00, 34.00, 44.00, 40.00]

    metrics = {
        "Hrate@1": hrate,
        "MRR": mrr,
        "F1": f1,
        "EM": em,
    }

    x = np.arange(len(methods))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    for offset, (metric_name, values) in zip(offsets, metrics.items()):
        ax.bar(x + offset, values, width, label=metric_name)

    ax.set_title("图5 MultiHop-RAG中检索指标与生成指标的差异", fontsize=15, fontweight="bold")
    ax.set_ylabel("指标数值")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 80)
    ax.legend()
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    ax.text(
        2.1,
        75,
        "SetSelection：Hrate@1、MRR最高\nListwise：F1、EM最高",
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
    )

    fig.tight_layout()
    save_fig(fig, "fig5_multihoprag_metric_gap.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-fonts", action="store_true", help="List available Chinese fonts and exit")
    args = parser.parse_args()
    if args.list_fonts:
        print(sorted({font.name for font in font_manager.fontManager.ttflist if any(tag in font.name for tag in ["CJK", "Hei", "Kai", "Song"])}))
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()
    fig1_rag_pipeline()
    fig2_rerank_strategy_compare()
    fig3_dataset_task_structure()
    fig4_f1_comparison()
    fig5_multihoprag_metric_gap()

    print(f"图片已生成到文件夹：{OUT_DIR.resolve()}")
    for path in sorted(OUT_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
