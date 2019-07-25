#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

def plot_network(network, position, value, **kwds):
    rag = LinearSegmentedColormap.from_list('RAG',
                                       ['xkcd:red',
                                        'xkcd:amber',
                                        'xkcd:green'])

    nx.draw(network, pos=position,
            node_size=75, node_color=value,
            cmap=rag,
            edge_color='xkcd:light grey',
            vmin=-1, vmax=1,
            **kwds)

def make_missing(y, p):
    y = np.array(y)
    n = len(y)
    miss_ix = np.random.choice(n, size=round(p*n), replace=False)
    miss = np.zeros(n, dtype=bool)
    miss[miss_ix] = True
    y[miss] = None
    return (y, miss)


def estimate(c, k, lmbda, U, y, miss):
    Lmbda = diags(1.0 / lmbda[0:k])
    U_obs = U[~miss,0:k]
    return U[:,0:k].dot(np.linalg.solve(U_obs.T.dot(U_obs) + c * Lmbda,
                         U_obs.T.dot(y[~miss])))

def MSE(f, y, miss):
    return np.mean(np.power(f[~miss] - y[~miss], 2))

def eigen(network, K):
    L = nx.laplacian_matrix(network)
    lmbda, U = eigsh(L.asfptype(), which='SM', k=K)
    lmbda[0] = lmbda[1]
    return (lmbda, U)

def main():
    tfl = nx.read_edgelist('data/lines', delimiter='|')
    n = nx.number_of_nodes(tfl)

    coordinates = {}
    zones = {}
    with open('data/stations.csv') as stations_file:
        station_reader = csv.reader(stations_file, delimiter='|')
        for row in station_reader:
            coordinates[row[0]] = [float(row[1]), float(row[2])]
            zones[row[0]] = row[3]

    plot_network(tfl, coordinates, 'xkcd:grey')

    K = 100
    lmbda, U = eigen(tfl, K)

    fig, axs = plt.subplots(4, 4)
    axs = axs.flatten()
    for k in range(16):
        plot_network(tfl, coordinates, np.sqrt(n) * U[:,k], ax=axs[k])

    max_fare = {"1": 7, "1/2": 7, "2": 7, "2/3": 7.6, "3": 8.2, "3/4": 9.15,
                "4": 10.1, "5": 12, "5/6": 12.4, "6": 12.8, "6/7": 13.4, "7": 14,
                "8": 16.5, "9": 18.3, "S": 18.3}
    y = []
    for station in tfl.nodes():
        value = (np.random.uniform())**(1/3) * max_fare[zones[station]]/6.4 - 1
        y.append(value)

    g = U.T.dot(y)

    y, miss = make_missing(y, 0.5)

    plt.figure()
    plot_network(tfl, coordinates, 'xkcd:light grey')
    plot_network(tfl, coordinates, y)

    f = estimate(0.0001, 50, lmbda, U, y, miss)
    plt.figure()
    plot_network(tfl, coordinates, f)

    P = nx.path_graph(n)
    kappa, V = eigen(P, K)
    z = V.dot(g)
    z[miss] = None

    err = []
    c_space = np.logspace(-10, 10)
    for c in c_space:
        h = estimate(c, 100, kappa, V, z, miss)
        err.append(MSE(h, V.dot(g), np.zeros(n, dtype=bool)))
    plt.figure()
    plt.scatter(np.log(c_space), err)


    h = estimate(1e-10, 100, kappa, V, z, miss)

    fig, axs = plt.subplots(4, 4)
    axs = axs.flatten()
    for k in range(16):
        axs[k].plot(range(n), np.sqrt(n) * V[:,k])
        axs[k].set_ylim([-1.25 * np.sqrt(2), 1.5 * np.sqrt(2)])
        axs[k].get_xaxis().set_ticks([])
        axs[k].get_yaxis().set_ticks([])

    plt.figure()
    plt.scatter(range(n), z)
    plt.plot(range(n), h)
    plt.show()

if __name__ == '__main__':
    main()
