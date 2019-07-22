# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:23:33 2019

@author: XL273XM
"""

from coordinates import coordinates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from zones import zones

tfl = nx.read_edgelist('lines', delimiter='|')
n = nx.number_of_nodes(tfl)

fig, ax = plt.subplots()
nx.draw_networkx(tfl, pos=coordinates, with_labels=False,
                 node_size=20, font_size=10, 
                 node_color='#333333',
                 edge_color='#999999')
ax.set_facecolor('#f0f0f0')
plt.show()

L = nx.laplacian_matrix(tfl)
K=100
lmbda, U = eigs(L.asfptype(), which='SM', k=K)

fig, axs = plt.subplots(4, 4)
axs = axs.flatten()
for k in range(16):
    nx.draw(tfl, pos=coordinates, with_labels=False,
            node_size=20, font_size=10, 
            cmap='RdYlGn', 
            vmin=-1.0/np.sqrt(n), vmax=1.0/np.sqrt(n),
            node_color=U[:,k], ax=axs[k])
plt.show()

fares = {1: 7, 1.5: 7, 2: 7, 2.5: 7.6, 3: 8.2, 3.5: 9.15, 4: 10.10, 4.5: 11.05, 
         5: 12, 5.5: 12.4, 6: 12.8, 6.5: 13.4, 7: 14, 7.5: 15.25, 8: 16.5, 
         9: 18.30, 9.5:22.92, 10: 27.55}

f = np.zeros(n)
i = 0
for station in tfl.nodes():
    zone = zones[station]
    if zone == 0:
        zone = 9
    fare = fares[zone]
    f[i] = fare * (np.random.uniform())**(1/3)
    i += 1

mis = np.random.choice(n, size=round(0.5*n))
mis.sort()
f[mis] = None
f_mis = np.zeros(n)
f_mis[mis] = 1
f_mis[f_mis==0] = None

fig, ax = plt.subplots()
nx.draw_networkx(tfl, pos=coordinates, with_labels=False,
                      node_size=20, node_color=f,
                      cmap='RdYlGn',
                      vmin=0, vmax=18.3,
                      edge_color='#cccccc')
nx.draw_networkx_nodes(tfl, pos=coordinates, with_labels=False, 
                       node_size=20, node_color=f_mis,
                       cmap='binary',
                       vmin=0, vmax=5)
plt.show()