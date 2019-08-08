import csv
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

def eigen_basis(network, k):
    """Computes the k smallest eigenvalues and corresponding eigenvectors of
    the Laplacian matrix of the network."""
    L = nx.laplacian_matrix(network)
    lmbda, U = eigsh(L.asfptype(), which='SM', k=k)
    return lmbda, U


def estimate_labels(lmbda, U, obs, c, k):
    """Estimates the underlying function on a network using the observations."""
    n = len(obs)
    miss = np.isnan(obs)
    lmbda[0] = lmbda[1]  # to make eigenvalues invertible
    Lmbda = diags(1.0 / lmbda[0:k])
    U_obs = U[~miss, 0:k]
    estimate = U[:,0:k].dot(np.linalg.solve(U_obs.T.dot(U_obs)
                            + c * Lmbda, U_obs.T.dot(obs[~miss])))
    return estimate

def rag_colormap():
    """Defines a red-amber-green colormap."""
    rag = LinearSegmentedColormap.from_list('RAG',
        ['xkcd:green', 'xkcd:amber', 'xkcd:red'])
    return rag


def read_station_data(file, **kwargs):
    """Reads stations coordinates and zone from a file."""
    coordinates = {}
    zones = {}
    with open(file) as stations_file:
        station_reader = csv.reader(stations_file, **kwargs)
        for row in station_reader:
            coordinates[row[0]] = [float(row[1]), float(row[2])]
            zones[row[0]] = row[3]
    return coordinates, zones


def plot_network(network, position, value, colormap=None, **kwargs):
    """Plots a network using fixed node size and edge color."""
    nx.draw(network, pos=position,
            node_size=75, node_color=value,
            cmap=colormap,
            edge_color='xkcd:light grey',
            **kwargs)


def tfl_zone_labels(zones, p):
    """Use zone data to make labels on the TfL network. An approximate fraction
    p of the labels will be missing."""

    def tfl_zone_value(zone):
        """Assigns a value to each zone."""
        try:
            return 10.0 - float(zone)
        except ValueError:
            if zone == "S":
                return 6.5
            else:
                zone_list = list(map(float, zone.split("/")))
                return 10 - sum(zone_list) / len(zone_list)

    zone_label = {}
    for station, zone in zones:
        missing = np.random.random() < p
        zone_label[station] = tfl_zone_value(zone) + np.random.normal()
