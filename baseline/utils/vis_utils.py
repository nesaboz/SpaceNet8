from builtins import range
from past.builtins import xrange

from math import sqrt, ceil
import numpy as np
import os
import sys


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid


def vis_grid(Xs):
    """ visualize a grid of images """
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A * H + A, A * W + A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y * H + y : (y + 1) * H + y, x * W + x : (x + 1) * W + x, :] = Xs[
                    n, :, :, :
                ]
                n += 1
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G


def vis_nn(rows):
    """ visualize array of arrays of images """
    N = len(rows)
    D = len(rows[0])
    H, W, C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N * H + N, D * W + D, C), Xs.dtype)
    for y in range(N):
        for x in range(D):
            G[y * H + y : (y + 1) * H + y, x * W + x : (x + 1) * W + x, :] = rows[y][x]
    # normalize to [0,1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G


from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["savefig.bbox"] = 'tight'


def plot_pil_images(imgs, titles, max_per_row=5, figsize=(5,5), 
                    save_path=None, remove=[], **imshow_kwargs):
    
    N = len(imgs)
    if titles:
        assert N == len(titles)
    else:
        titles = ['']*len(imgs)
    
    if N <= max_per_row:
        imgs = [imgs]
        titles = [titles]
    else:
        # convert 1D imgs to 2D dimgs
        dimgs = []
        dtitles = []
        while len(imgs) > max_per_row:
            dimgs.append(imgs[:max_per_row])
            dtitles.append(titles[:max_per_row])
            imgs = imgs[max_per_row:]
            titles = titles[max_per_row:]
        dimgs.append(imgs)
        dtitles.append(titles)
        imgs = dimgs
        titles = dtitles
        
        titles = [''] * N
     
    if N <= max_per_row:
        imgs = [imgs]
        titles = [titles]
    else:
        # convert 1D imgs to 2D dimgs
        dimgs = []
        dtitles = []
        while len(imgs) > max_per_row:
            dimgs.append(imgs[:max_per_row])
            dtitles.append(titles[:max_per_row])
            imgs = imgs[max_per_row:]
            titles = titles[max_per_row:]
        dimgs.append(imgs)
        dtitles.append(titles)
        imgs = dimgs
        titles = dtitles
        
    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)

    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            if num_rows == 1:
                ax = axs[col_idx]
            else:
                ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_title(titles[row_idx][col_idx])
     
    # delete unused axes
    for i in remove + list(range(N, num_rows*num_cols)):
        if num_rows == 1:
            ax = axs[i]
        else:
            ax = axs[i//num_cols, i%num_cols]
        ax.set_axis_off()
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path,
                    dpi=300,
                    bbox_inches='tight'
                    # bbox_inches=Bbox.from_extents(-0.2, -0.2, 8, 4)
        )
        print(f'Saved to {save_path}')
    
    plt.show()