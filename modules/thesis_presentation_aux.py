import csv
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import numpy as np


def eigen_basis(network):
    """Computes the k smallest eigenvalues and corresponding eigenvectors of
    the Laplacian matrix of the network."""
    L = nx.laplacian_matrix(network)
    lmbda, U = np.linalg.eigh(L.todense())
    return lmbda, U


def estimate_labels(lmbda, U, obs, c, k):
    """Estimates the underlying function on a network using the observations."""
    n = len(obs)
    miss = np.isnan(obs)
    lmbda[0] = lmbda[1]  # to make eigenvalues invertible
    Lmbda = np.diag(1.0 / lmbda[0:k])
    U_obs = U[~miss,0:k]
    estimate = U[:,0:k].dot(np.linalg.solve(U_obs.T.dot(U_obs)
        + c * Lmbda, U_obs.T.dot(obs[~miss]).T))
    return np.ravel(estimate)


def estimate_rough(obs):
    """Interpolates the observed data on a network."""
    x = np.linspace(0, 1, len(obs))
    miss = np.isnan(obs)
    x[~miss]
    estimate = np.interp(x, x[~miss], obs[~miss])
    return estimate

def estimate_smooth(obs):
    """Take the average of the observed data on a network."""
    estimate = np.nanmean(obs) * np.ones(len(obs))
    return estimate


def line_example(tfl, line, tfl_labels):
    """Gets the line labels from the larger TfL network for a certain line."""
    line_labels = []
    i = 0
    for station in tfl.nodes:
        if station in line:
            line_labels.append(tfl_labels[i])
        i = i + 1
    return np.array(line_labels)


def plot_network(network, position, value, colormap=None, **kwargs):
    """Plots a network using fixed node size and edge color."""
    nx.draw(network, pos=position,
            node_size=75, node_color=value,
            cmap=colormap,
            edge_color='xkcd:light grey',
            **kwargs)


def plot_path_graph(ax, y, color='xkcd:blue', **kwargs):
    ax.plot(y, linewidth=2, c=color, **kwargs)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


def project(target, basis, k):
    """Project the target on the first k terms of the basis."""
    return np.ravel(basis[:, 0:k].dot(basis[:, 0:k].T).dot(target))


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


def rough_example(n):
    """Returns n function values of a rough function."""
    return np.random.normal(size=n)


def smooth_example(basis, k):
    """Returns values of a smooth function that is the composition of k
    basis function."""
    U = basis[:, 0:k]
    n = U.shape[0]
    return np.ravel(U.dot(U.T).dot(np.random.normal(size=n)))


def scatter_path_graph(ax, y, **kwargs):
    ax.scatter(range(len(y)), y,  marker='.', c='xkcd:black', **kwargs)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


def tfl_zone_labels(tfl, zones, p):
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

    zone_label = []
    for station in tfl.nodes:
        missing = np.random.random() < p
        if missing:
            zone_label.append(None)
        else:
            zone_label.append(tfl_zone_value(zones[station])
            + np.random.normal())
    return np.array(zone_label, dtype=float)
