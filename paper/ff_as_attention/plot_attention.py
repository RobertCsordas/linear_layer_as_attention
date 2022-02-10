from optparse import Option
from matplotlib import image
import lib
import wandb
import json
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import math
from dataclasses import dataclass
import imageio


@dataclass
class Source:
    name: str
    runid: str
    plotroot: str
    extra_height_score: List[float]
    extra_height_sum: List[float]
    legend: Optional[List[str]]
    score_legend_loc: Optional[str] = None
    sum_legend_loc: Optional[str] = None
    score_legend_ncol: Optional[int] = None
    
os.makedirs("out", exist_ok = True)

sources = [
    Source("mnist_single_6", "username/ff_as_attention/0ze73yjx", "correct/test_mnist/0", [0.07, 0, 0], [0, 0, 0], None),
    Source("mnist_fmnist_layer", "username/ff_as_attention/lmyo6bgh", "correct/test_fmnist/2", [0.07, 0, 0.07], [0, 0.05, 0.05], ["MNIST", "FashionMNIST"]),
    Source("sequential_misclassified_", "username/ff_as_attention/2oao52kr", "misclassified/test_mnist/0", [0, 0.07, 0.07], [0.05, 0, 0], ["MNIST", "FashionMNIST"], 'lower center', 'upper right', score_legend_ncol=2),
    Source("mnist_correct", "username/ff_as_attention/as77vdkr", "correct/test_mnist/0", [0.07, 0, 0.00], [0, 0, 0], None),
    Source("mixed_correct_1", "username/ff_as_attention/9nu7vsq9", "correct/test_fmnist/5", [0, 0, 0], [0, 0.04, 0.04], ["MNIST", "FashionMNIST"]),
    Source("mixed_correct_2", "username/ff_as_attention/9nu7vsq9", "correct/test_fmnist/0", [0, 0.07, 0.07], [0, 0, 0.0], ["MNIST", "FashionMNIST"], None, 'upper left'),
    Source("sequential_correct_1", "username/ff_as_attention/ne0jro2c", "correct/test_fmnist/0", [0, 0, 0], [0, 0, 0], ["MNIST", "FashionMNIST"]),
    Source("sequential_correct_2", "username/ff_as_attention/0lajghya", "correct/test_mnist/5", [0, 0, 0], [0, 0., 0.], ["MNIST", "FashionMNIST"], 'lower left', score_legend_ncol=2),
]

def find_x_groups(x: List[float], y: List[float]) -> List[Tuple[float, float]]:
    res = []
    x_start = None
    for i, f in enumerate(x):
        if math.isnan(y[i]):
            res.append((x_start, x[i-1]))
            x_start = None
        elif x_start is None:
            x_start = f

    if x_start is not None:
        res.append((x_start, x[-1]))

    return res

def plot_image(name):
    img = run.file(run.summary[name].path).download(replace=True)
    img = imageio.imread(img.name)

    fig = plt.figure(figsize=[2,2])
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    return fig


def plot_input(run, src: Source):
    fig = plot_image(f"analysis/{src.plotroot}/input")
    fig.savefig(f"out/{src.name}_input.pdf", bbox_inches='tight', pad_inches=0.01)


def plot_top3(run, src: Source):
    print(f"Example figure for {src.name}")
    print("\\begin{figure*}[ht]")

    for l in range(3):
        for i in range(3):
            fig = plot_image(f"analysis/{src.plotroot}/per_update/layers.{l}/closest_training_{i}")
            fig.savefig(f"out/{src.name}_layer_{l}_closest_{i}.pdf", bbox_inches='tight', pad_inches=0.01)

            print(f"    \\subfloat[layer-{l} top-{i+1}]{{")
            print( "        \\centering")
            print(f"        \\includegraphics[width=.2\\linewidth]{{figures/{src.name}_layer_{l}_closest_{i}.pdf}}")
            print( "    }")

        if l != 2:
            print("\\\\")
    name_latex = src.name.replace('_', '\\_')
    print(f"    \\caption{{Top examples for {name_latex}}}")
    print("\\end{figure*}")

def plot_attention_scores(run, src: Source):
    for l in range(3):
        f = run.file(run.summary[f"analysis/{src.plotroot}/per_update/layers.{l}/magic_class_plot"].path).download(replace=True)

        d = json.loads(f.read())

        x = [a['x'] for a in d["data"]]
        y = [a['y'] for a in d["data"]]

        order = list(sorted(range(len(x)), key=lambda i: x[i][0]))
        x = [x[o] for o in order]
        y = [y[o] for o in order]

        min_x = min([min(a) for a in x])
        max_x = max([max(a) for a in x])
        min_y = min([min(a) for a in y])
        max_y = max([max(a) for a in y])

        groups = sum([find_x_groups(a, b) for a, b in zip(x, y)], [])
        ticks = [(g[0] + g[1])/2 for g in groups]

        fig = plt.figure(figsize=[5,2.0+src.extra_height_score[l]])
        for a, b in zip(x,y):
            plt.plot(a, b, linewidth=1, marker="o", markersize=1.5)
        plt.xticks(ticks, [str(i % 10) for i in range(len(ticks))])
        # plt.xlim(min(x) max(x)+1)
        plt.ylim(min_y-20, max_y+20)
        plt.xlim(min_x-40, max_x+40)
        plt.vlines([g[1]+1 for g in groups[:-1]], min_y-20, max_y+20, colors="gray", linewidth=1, alpha=0.25, zorder=0)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

        if l==0 and src.legend is not None:
            plt.legend(src.legend, loc=src.score_legend_loc, ncol=src.score_legend_ncol or 1)

        fig.savefig(f"out/{src.name}_layer_{l}.pdf", bbox_inches='tight', pad_inches=0.02)

def plot_score_sum(run, src: Source):
    for l in range(3):
        f = run.file(run.summary[f"analysis/{src.plotroot}/per_update/layers.{l}/total_score_per_class"].path).download(replace=True)
        d = json.loads(f.read())

        x = [a['x'] for a in d["data"]]
        y = [a['y'] for a in d["data"]]

        order = list(sorted(range(len(x)), key=lambda i: x[i][0]))
        x = [x[o] for o in order]
        y = [y[o] for o in order]

        n_total = sum(len(a) for a in x)


        fig = plt.figure(figsize=[5,2+src.extra_height_sum[l]])
        for a, b in zip(x,y):
            plt.bar(a, b)

        plt.xticks(list(range(n_total)), [str(i % 10) for i in range(n_total)])
        if l==0 and src.legend is not None:
            plt.legend(src.legend, loc=src.sum_legend_loc or 'lower left', ncol=2)

        fig.savefig(f"out/{src.name}_layer_{l}_sum.pdf", bbox_inches='tight', pad_inches=0.02)


api = wandb.Api()

for src in sources:
    print(f"Generating {src.name}")
    run = api.run(src.runid)
    plot_input(run, src)
    plot_top3(run, src)
    plot_attention_scores(run, src)
    plot_score_sum(run, src)
