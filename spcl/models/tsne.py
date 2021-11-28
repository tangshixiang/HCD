# That's an impressive list of imports.
import numpy as np
from sklearn.manifold import TSNE

RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

import json
import pickle
import pdb


# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})


center = 0


def _indexing(labels, ignore_labels=[-1]):
  label_to_index = {}
  for idx, lab in enumerate(labels):
    if lab in ignore_labels:
      continue
    if lab in label_to_index:
      label_to_index[lab].append(idx)
    else:
      label_to_index[lab] = [idx]
  return label_to_index


def label2ordered_index(y):
    unq = np.unique(y)
    lb_transform = {x: c for c, x in enumerate(unq)}
    for i in range(len(y)):
        y[i] = lb_transform[y[i]]
    return y


def scatter(x, colors):
    num_colors =  len(np.unique(colors))
    print('num colors:', num_colors)

    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", num_colors))
    palette = np.vstack([palette, np.float32([[0, 0.3, 0]])])

    # We create a scatter plot.
    f = plt.figure(figsize=(16, 16))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=3, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(len(colors)):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=8)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=3, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


if __name__ == '__main__':
    name = 'ms1m_part1_test'

    X = np.fromfile('features.bin'.format(name), dtype=np.float32).reshape(-1, 2048)
    y = np.loadtxt('label.txt'.format(name), dtype=np.int)

    lb2idx = _indexing(y)

    dist = np.dot(X, X[center])
    nbr = np.argsort(-dist)[:100]
    nbr_label = np.unique(y[nbr])
    C = len(nbr_label)

    index = []
    for i in nbr_label:
        index.extend(lb2idx[i])
    print('inst:', len(index))
    X = X[index]
    y = y[index]
    y = label2ordered_index(y)


    proj = TSNE(random_state=RS).fit_transform(X)
    scatter(proj, y)
    plt.savefig('out.png', dpi=120)
