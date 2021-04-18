import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

import torch
import torchvision.utils as utils
import torch.nn.functional as F
import torch.nn as nn


def select_data_dir(data_dir='../data'):
    data_dir = '/coursedata' if os.path.isdir('/coursedata') else data_dir
    print('The data directory is %s' % data_dir)
    return data_dir


def get_validation_mode():
    try:
        return bool(os.environ['NBGRADER_VALIDATING'])
    except:
        return False


def save_model(model, filename, confirm=True):
    if confirm:
        try:
            save = input('Do you want to save the model (type yes to confirm)? ').lower()
            if save != 'yes':
                print('Model not saved.')
                return
        except:
            raise Exception('The notebook should be run or validated with skip_training=True.')

    torch.save(model.state_dict(), filename)
    print('Model saved to %s.' % (filename))


def load_model(model, filename, device):
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    model.eval()


def plot_images(images, ncol=12, figsize=(8,8), cmap=plt.cm.Greys, clim=[0,1]):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    grid = utils.make_grid(images, nrow=ncol, padding=0, normalize=False).cpu()
    ax.imshow(grid[0], cmap=cmap, clim=clim)
    display.display(fig)
    plt.close(fig)


def plot_generated_samples(samples, ncol=12):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.axis('off')
    ax.imshow(
        np.transpose(
            utils.make_grid(samples, nrow=ncol, padding=0, normalize=True).cpu(),
            (1,2,0)
        )
     )
    display.display(fig)
    plt.close(fig)


def show_proba(proba, r, c, ax):
    """Creates a matshow-style plot representing the probabilites of the nine digits in a cell.
    
    Args:
      proba of shape (9): Probabilities of 9 digits.
    """
    cm = plt.cm.Reds
    ix = proba.argmax()
    if proba[ix] > 0.9:
        px, py = c+0.5, r+0.5
        ax.text(px, py, ix.item(), ha='center', va='center', fontsize=24)
    else:
        for d in range(9):
            dx = dy = 1/6
            px = c + dx + (d // 3)*(2*dx)
            py = r + dy + (d % 3)*(2*dy)
            p = proba[d]
            ax.fill(
                [px-dx, px+dx, px+dx, px-dx, px-dx], [py-dy, py-dy, py+dy, py+dy, py-dy],
                #color=[p, 1-p, 1-p]
                color=cm(int(p*256))
            )
            ax.text(px, py, d, ha='center', va='center', fontsize=8)    

