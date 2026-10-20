#!/usr/bin/env python3
"""Draw the Token Bank v2 method architecture figure with proper CJK font support."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Pick a CJK-capable system font available on macOS.
CANDIDATES = ['Arial Unicode MS', 'Heiti SC', 'Hiragino Sans GB', 'PingFang SC', 'SimHei']
from matplotlib import font_manager
available = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((c for c in CANDIDATES if c in available), 'DejaVu Sans')

plt.rcParams['font.family'] = FONT
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

COLORS = {
    'agent': '#dbeafe',      # blue-100
    'agent_border': '#3b82f6', # blue-500
    'input': '#dcfce7',      # green-100
    'input_border': '#22c55e', # green-500
    'action': '#fef9c3',     # yellow-100
    'action_border': '#eab308', # yellow-500
    'tool': '#fee2e2',       # red-100
    'tool_border': '#ef4444', # red-500
    'bank': '#f3f4f6',       # gray-100
    'bank_border': '#6b7280', # gray-500
    'arrow': '#374151',      # gray-700
}


def add_box(ax, x, y, w, h, text, color, edge, fontsize=11, bold=False, radius=0.02):
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle=f"round,pad=0.02,rounding_size={radius}",
                         facecolor=color, edgecolor=edge, linewidth=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, wrap=True)
    return box


def add_arrow(ax, x1, y1, x2, y2, color=COLORS['arrow'], style='->', lw=1.5, 
              connectionstyle="arc3,rad=0"):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle=connectionstyle))


def main():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Title
    fig.suptitle('Token Bank v2 方法架构图', fontsize=20, fontweight='bold', y=0.96)

    # Top agent pipeline
    add_box(ax, 1.4, 8.6, 2.0, 0.8, '输入任务\n(question + 附件)', '#f3e8ff', '#a855f7', fontsize=10)
    add_box(ax, 4.0, 8.6, 1.8, 0.8, 'Planner\n(预算系数 3)', COLORS['agent'], COLORS['agent_border'], fontsize=10, bold=True)
    add_box(ax, 6.4, 8.6, 1.8, 0.8, 'Executor\n(预算系数 2)', COLORS['agent'], COLORS['agent_border'], fontsize=10, bold=True)
    add_box(ax, 8.8, 8.6, 1.8, 0.8, 'Reviewer\n(预算系数 1)', COLORS['agent'], COLORS['agent_border'], fontsize=10, bold=True)

    add_arrow(ax, 2.4, 8.6, 3.1, 8.6)
    add_arrow(ax, 4.9, 8.6, 5.5, 8.6)
    add_arrow(ax, 7.3, 8.6, 7.9, 8.6)

    # Central TokenBankV2
    add_box(ax, 6.4, 6.3, 4.6, 1.4,
            'TokenBankV2\n总配额 B → 按 Agent 优先级拆分 q_i',
            COLORS['bank'], COLORS['bank_border'], fontsize=13, bold=True, radius=0.06)

    # Arrows from agents down to bank
    add_arrow(ax, 4.0, 8.2, 4.7, 7.0, color=COLORS['agent_border'], style='->', lw=2)
    add_arrow(ax, 6.4, 8.2, 6.4, 7.0, color=COLORS['agent_border'], style='->', lw=2)
    add_arrow(ax, 8.8, 8.2, 8.1, 7.0, color=COLORS['agent_border'], style='->', lw=2)
    ax.text(4.2, 7.6, '额度请求', fontsize=9, color=COLORS['agent_border'], ha='left')
    ax.text(8.0, 7.6, '额度请求', fontsize=9, color=COLORS['agent_border'], ha='right')

    # Reviewer -> Executor feedback loop
    add_arrow(ax, 9.7, 8.2, 9.7, 6.3, color='#9ca3af', style='->', lw=1.2,
              connectionstyle="arc3,rad=0.3")
    ax.text(10.2, 7.25, '重审触发', fontsize=9, color='#6b7280', rotation=90, va='center')

    # Left inputs
    inputs = [
        (1.5, 5.1, '历史成功率\n(γ 滚动 plan + runtime)'),
        (1.5, 3.9, '当前任务难度\n(theta 估计)'),
        (1.5, 2.7, 'Agent 优先级\n(credit_rate=1.5)'),
        (1.5, 1.5, 'KV-cache 长度\n(压缩阈值判断)'),
    ]
    for x, y, txt in inputs:
        add_box(ax, x, y, 2.6, 0.9, txt, COLORS['input'], COLORS['input_border'], fontsize=10)
        add_arrow(ax, 2.8, y, 4.1, 6.0, color=COLORS['input_border'], style='->', lw=1.5)
    ax.text(1.5, 6.0, '输入信息', fontsize=11, fontweight='bold', ha='center')

    # Right actions
    add_box(ax, 11.8, 5.1, 2.6, 1.0, '配额再分配 OPL\n(Thompson Sampling)',
            COLORS['action'], COLORS['action_border'], fontsize=10)
    add_box(ax, 11.8, 3.6, 2.6, 1.0, '信用评分 CreditScorer\n(rogue-only 风控)',
            COLORS['action'], COLORS['action_border'], fontsize=10)
    add_arrow(ax, 8.7, 6.0, 10.5, 5.1, color=COLORS['action_border'], style='->', lw=1.5)
    add_arrow(ax, 8.7, 5.5, 10.5, 3.6, color=COLORS['action_border'], style='->', lw=1.5)
    ax.text(11.8, 6.0, '输出动作', fontsize=11, fontweight='bold', ha='center')

    # Bottom tool / result flow
    add_box(ax, 6.4, 1.5, 3.4, 1.0, '代码执行器\n(```python 自动执行)',
            COLORS['tool'], COLORS['tool_border'], fontsize=10)
    add_arrow(ax, 6.4, 5.6, 6.4, 2.0, color=COLORS['tool_border'], style='->', lw=2)
    ax.text(5.2, 3.8, '触发', fontsize=9, color=COLORS['tool_border'])

    add_box(ax, 10.8, 1.5, 2.6, 1.0, '结果回灌\n(更新上下文 / 触发重审)',
            '#f3e8ff', '#a855f7', fontsize=10)
    add_arrow(ax, 8.1, 1.5, 9.5, 1.5, color='#a855f7', style='->', lw=1.5)

    # Legend
    legend_items = [
        ('Agent 模块', COLORS['agent'], COLORS['agent_border']),
        ('输入信息', COLORS['input'], COLORS['input_border']),
        ('输出动作', COLORS['action'], COLORS['action_border']),
        ('工具/回灌', '#f3e8ff', '#a855f7'),
    ]
    for i, (label, fc, ec) in enumerate(legend_items):
        lx = 1.0 + i * 2.2
        ly = 0.35
        rect = FancyBboxPatch((lx, ly), 0.4, 0.25, boxstyle='round,pad=0.01,rounding_size=0.02',
                              facecolor=fc, edgecolor=ec, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(lx + 0.55, ly + 0.125, label, va='center', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    out_path = '/Users/doby/Desktop/tokenbank/token_bank_v2_method.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved method figure to {out_path} using font: {FONT}')


if __name__ == '__main__':
    main()
