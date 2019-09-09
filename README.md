# Nonparametric Bayesian label prediction on a graph

This notebook contains plots and parts of the storyline for the presentation of the doctoral thesis "Nonparametric Bayesian label prediction on a graph" that is comprehensible to non-specialists. It includes an analysis of simulated station traffic data on the Transport for London network (tube, DLR, cable car, overground, TfL rail and tram connections). The edges in the network are actual connections and stations that are within walking distance as per the [TfL network](http://content.tfl.gov.uk/large-print-tube-map.pdf) (May 2019).

* The data folder contains the edge list of the TfL network and station coordinates and zones in .csv format. The Lines.xlsx file contains the edge list with a separate sheet per transport line, it is not used in our script, but included for readability.
* The modules folder contains thesis_presentation_aux.py which is a module with auxiliary functions used in this notebook.
* This notebook contains the storyline, plots and example analysis of the TfL network.
