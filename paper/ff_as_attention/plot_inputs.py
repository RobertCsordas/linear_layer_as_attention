import lib
import sys
import os
import matplotlib.pyplot as plt

os.makedirs("out", exist_ok=True)
sys.path.insert(0, os.path.dirname(__file__)+"/../..")

import dataset

inputs = [5938]

ds = dataset.image.PermutedMNIST(0, "test")
for i in inputs:
    fig = plt.figure(figsize=[2,2])
    plt.imshow(ds.unnormalize(ds[i]["image"])[0], cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    fig.savefig(f"out/mnist_{i}.pdf", bbox_inches='tight', pad_inches=0.01)


inputs = [981]
ds = dataset.image.FashionMNIST(0, "test")
for i in inputs:
    fig = plt.figure(figsize=[2,2])
    plt.imshow(ds.unnormalize(ds[i]["image"])[0], cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    fig.savefig(f"out/fmnist_{i}.pdf", bbox_inches='tight', pad_inches=0.01)