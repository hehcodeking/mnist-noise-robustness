"""Rebuild all report figures and numeric summary from measured outputs."""
import csv
import json
import os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache/matplotlib'))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from data import ROOT, load_data, normalized, gaussian

FONT = Path('/System/Library/Fonts/STHeiti Medium.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
else:
    # Linux users can install Noto Sans CJK SC before regenerating Chinese plots.
    plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.unicode_minus': False, 'savefig.dpi': 240,
                     'axes.labelcolor': '#222222', 'text.color': '#222222'})
NAMES = {'linear': '线性基线', 'cnn_clean': 'CNN 普通训练',
         'cnn_noise015': 'CNN 增强 0.15', 'cnn_noise030': 'CNN 增强 0.30'}
COLORS = {'linear': '#9299A1', 'cnn_clean': '#335C81',
          'cnn_noise015': '#D0983C', 'cnn_noise030': '#1E7B68'}


def save(fig, name):
    folder = ROOT / 'figures'
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / (name + '.png'), bbox_inches='tight', facecolor='white')
    fig.savefig(folder / (name + '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    config = json.loads((ROOT / 'config.json').read_text())
    with (ROOT / 'results/metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    summary = {}
    for name in NAMES:
        summary[name] = {}
        for corruption in sorted({r['corruption'] for r in rows}):
            records = [r for r in rows if r['condition'] == name and r['corruption'] == corruption]
            item = {}
            for key in ['accuracy', 'macro_f1', 'training_seconds', 'best_epoch', 'parameters']:
                vals = np.array([float(r[key]) for r in records])
                item[key] = {'mean': float(vals.mean()), 'sd': float(vals.std(ddof=1)), 'values': vals.tolist()}
            summary[name][corruption] = item
    (ROOT / 'results/summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(figsize=(6.65, 2.95), layout='constrained')
    for name in NAMES:
        means = [summary[name][f'gaussian_{s:.1f}']['accuracy']['mean'] * 100 for s in config['test_sigmas']]
        sd = np.array([summary[name][f'gaussian_{s:.1f}']['accuracy']['sd'] * 100 for s in config['test_sigmas']])
        ax.plot(config['test_sigmas'], means, marker='o', ms=4, lw=1.8, color=COLORS[name], label=NAMES[name])
        ax.fill_between(config['test_sigmas'], np.array(means)-sd, np.array(means)+sd, color=COLORS[name], alpha=.12, linewidth=0)
    ax.set(xlabel='测试高斯噪声标准差 σ', ylabel='准确率（%）', xticks=config['test_sigmas'], ylim=(35, 101))
    ax.grid(axis='y', color='#e8ebef', lw=.7)
    ax.legend(loc='lower left', frameon=False, fontsize=8.5, ncols=2)
    save(fig, 'robustness_curve')

    fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.6), layout='constrained')
    for name in ['cnn_clean', 'cnn_noise015', 'cnn_noise030']:
        with (ROOT / f'results/history_{name}_11.csv').open() as stream:
            history = list(csv.DictReader(stream))
        epochs = [int(r['epoch']) for r in history]
        axes[0].plot(epochs, [float(r['train_loss']) for r in history], color=COLORS[name], label=NAMES[name], lw=1.7)
        axes[1].plot(epochs, [float(r['selection_score'])*100 for r in history], color=COLORS[name], label=NAMES[name], lw=1.7)
    axes[0].set(xlabel='训练轮数', ylabel='训练交叉熵')
    axes[1].set(xlabel='训练轮数', ylabel='验证集平均准确率（%）')
    for ax in axes:
        ax.set_xticks([1, 4, 8, 12]); ax.grid(axis='y', color='#e8ebef', lw=.7)
    axes[0].legend(frameon=False, fontsize=7.5)
    save(fig, 'learning_curves')

    _, _, raw, labels = load_data()
    x = normalized(raw)
    noise = np.random.default_rng(config['test_noise_seed']).standard_normal(x.shape, dtype=np.float32)
    x_noisy = gaussian(x, .3, noise)
    preds = np.load(ROOT / 'results/predictions.npz')
    a = preds['cnn_clean_11_gaussian_0.3']
    b = preds['cnn_noise030_11_gaussian_0.3']
    recovered = np.flatnonzero((a != labels) & (b == labels))[:4]
    failed = np.flatnonzero(b != labels)[:2]
    ids = np.concatenate([recovered, failed])
    # Cases are the first qualifying official test indices, rather than manually selected images.
    fig, axes = plt.subplots(2, 6, figsize=(6.65, 2.7), layout='constrained')
    for j, i in enumerate(ids):
        axes[0,j].imshow(x[i,0], cmap='gray', vmin=0, vmax=1)
        axes[1,j].imshow(x_noisy[i,0], cmap='gray', vmin=0, vmax=1)
        axes[0,j].set_title(f'#{i} 真值 {labels[i]}', fontsize=8)
        axes[1,j].set_title(f'普通 {a[i]}  增强 {b[i]}', fontsize=8,
                            color='#1E7B68' if b[i] == labels[i] else '#B04A40')
        for row in range(2):
            axes[row,j].set_xticks([]); axes[row,j].set_yticks([])
            for spine in axes[row,j].spines.values(): spine.set_visible(False)
    save(fig, 'cases')
    (ROOT / 'results/case_indices.json').write_text(json.dumps(ids.tolist()))

    fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.65), layout='constrained')
    for ax, pred, label in zip(axes,[a,b],['CNN 普通训练','CNN 增强 0.30']):
        cm = np.zeros((10,10), int); np.add.at(cm,(labels,pred),1)
        normalized_cm = cm / cm.sum(axis=1,keepdims=True) * 100
        im=ax.imshow(normalized_cm, vmin=0, vmax=100, cmap='Blues')
        ax.set(xlabel='预测类别',ylabel='真实类别',xticks=range(10),yticks=range(10),title=label)
        ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axes, label='占真实类别比例（%）', shrink=.83, pad=.025)
    save(fig, 'confusion')


if __name__ == '__main__':
    main()
