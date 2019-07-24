#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs

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
    plt.show()

    L = nx.laplacian_matrix(tfl)
    K = 100
    lmbda, U = eigs(L.asfptype(), which='SM', k=K)

    fig, axs = plt.subplots(4, 4)
    axs = axs.flatten()
    for k in range(16):
        plot_network(tfl, coordinates, np.sqrt(n) * U[:,k], ax=axs[k])
    plt.show()

    mapping = {"1":7, "1/2":7, "2":7, "2/3":7.6, "3":8.2, "3/4":9.15, "4":10.1,
               "5":12, "5/6":12.4, "6":12.8,
               "6/7":13.4,
               "7":14,
               "8":16.5, "9":18.3, "S":18.3}
    f = []
    for station in tfl.nodes():
        value = (np.random.uniform())**(1/3) * mapping[zones[station]]/6.4 - 1
        f.append(value)
    f = np.array(f)

    miss = np.random.choice(n, size=round(0.5*n), replace=False)
    f[miss] = None

    plt.figure()
    plot_network(tfl, coordinates, 'xkcd:light grey')
    plot_network(tfl, coordinates, f)
    plot.show()

if __name__ == '__main__':
    main()
