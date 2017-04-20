#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('extraction')
parser.add_argument('-o', '--output', required=False)
args = parser.parse_args()

data = np.load(args.extraction)

fig, axis = plt.subplots()
axis.imshow(data.T, cmap='viridis', origin='lower')
axis.set(ylabel='Time')
fig.tight_layout()
if args.output is not None:
    fig.savefig(args.output)
else:
    plt.show()
