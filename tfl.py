#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

def rag_colormap():
    rag = LinearSegmentedColormap.from_list('RAG',
                                            ['xkcd:green',
                                            'xkcd:amber',
                                            'xkcd:red'])
    return rag

def plot_network(network, position, value, **kwds):
    rag = rag_colormap()
    nx.draw(network, pos=position,
            node_size=75, node_color=value,
            cmap=rag,
            edge_color='xkcd:light grey',
            vmin=-4, vmax=4,
            **kwds)

def plot_path_graph(ax, x, f, color='xkcd:blue', **kwargs):
    ax.plot(x, f, linewidth=2, c=color, **kwargs)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_ylim([-5, 5])

def scatter_path_graph(ax, x, y, **kwargs):
    ax.scatter(x, y,  marker='.', c='xkcd:black', **kwargs)

def make_missing(y, p):
    n = len(y)
    miss_ix = np.random.choice(n, size=round(p*n), replace=False)
    miss = np.zeros(n, dtype=bool)
    miss[miss_ix] = True
    y[miss] = None
    return (y, miss)

def estimate(c, k, lmbda, U, y, miss):
    Lmbda = diags(1.0 / lmbda[0:k])
    U_obs = U[~miss,0:k]
    return U[:,0:k].dot(np.linalg.solve(U_obs.T.dot(U_obs) +
                        c * Lmbda, U_obs.T.dot(y[~miss])))

def MSE(f, y):
    return np.square(np.subtract(f, y)).mean()

def eigen(network, K):
    L = nx.laplacian_matrix(network)
    lmbda, U = eigsh(L.asfptype(), which='SM', k=K)
    lmbda[0] = lmbda[1]
    return (lmbda, U)

def optimize(lmbda, U, y, miss, f0):
    n = len(y)
    minimum = 1e10
    c_space = np.logspace(-8, 8)
    k_space = np.arange(2, 128, 1)
    for k in k_space:
        for c in c_space:
            f = estimate(c, k, lmbda, U, y, miss)
            error = MSE(f, f0)
            if error < minimum:
                minimum = error
                c_opt, k_opt = c, k
    return (c_opt, k_opt)

def main():
    # Read TfL network
    tfl = nx.read_edgelist('data/lines', delimiter='|')
    n = nx.number_of_nodes(tfl)

    # Read station and zone coordinates
    coordinates = {}
    zones = {}
    with open('data/stations.csv') as stations_file:
        station_reader = csv.reader(stations_file, delimiter='|')
        for row in station_reader:
            coordinates[row[0]] = [float(row[1]), float(row[2])]
            zones[row[0]] = row[3]

    # Plot TfL network
    plt.figure()
    plot_network(tfl, coordinates, 'xkcd:grey')

    # Path graph example
    x = np.linspace(0, 1, num=n)
    h0 = np.sin(12 * (x + 0.2)) / (x + 0.2)
    z = h0 + np.random.normal(size=n)
    z, miss = make_missing(z, 0.5)

    # Plot path graph example function and data
    fig, ax = plt.subplots()
    plot_path_graph(ax, x, h0)
    scatter_path_graph(ax, x, z)

    # Extract example coefficients
    P = nx.path_graph(n)
    K = 256
    kappa, V = eigen(P, K)
    g0 = V.T.dot(h0)

    # TfL network function using example coefficients
    lmbda, U = eigen(tfl, K)
    f0 = U.dot(g0)
    y = f0 + np.random.normal(size=n)
    y[miss] = None

    # Plot TfL example function
    plt.figure()
    plot_network(tfl, coordinates, 'xkcd:light grey')
    plot_network(tfl, coordinates, y)

    # Plot path graph eigenfunctions
    fig, axs = plt.subplots(4, 4)
    axs = axs.flatten()
    for k in range(16):
        plot_path_graph(axs[k], x, 4.0 * np.sqrt(0.5 * n) * V[:,k])

    # Plot TfL network eigenfunctions
    fig, axs = plt.subplots(4, 4)
    axs = axs.flatten()
    for k in range(16):
        plot_network(tfl, coordinates, 5 * np.sqrt(n) * U[:,k], ax=axs[k])

    # Path graph regression
    c_opt, k_opt = optimize(kappa, V, z, miss, h0)
    print(c_opt, k_opt)
    h = estimate(c_opt, k_opt, kappa, V, z, miss)

    # Plot path graph estimate
    fig, ax = plt.subplots()
    plot_path_graph(ax, x, h0)
    scatter_path_graph(ax, x, z)
    plot_path_graph(ax, x, h, color='xkcd:orange')

    # TfL network regression
    c_opt, k_opt = optimize(lmbda, U, y, miss, f0)
    print(c_opt, k_opt)
    f = estimate(c_opt, k_opt, lmbda, U, y, miss)

    # Plot TfL estimate
    plt.figure()
    plot_network(tfl, coordinates, f)

if __name__ == '__main__':
    main()
    plt.show()
