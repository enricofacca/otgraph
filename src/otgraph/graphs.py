import numpy as np
import scipy
from copy import deepcopy as cp


class Graph:
    """
    This class contains the inputs
    of problem GraphDmk. Namely
    min |v|^{q>=1} : A v = rhs
    with A signed incidence matrix of graph G
    """

    def __init__(self, topol, weight=None):
        """
        Constructor from raw data

        Args:
        topol:  (2,n_edge) integer np.array with node conenctivity
                The order define the orientation
        weight: (n_edge) real np.array with weigth associate to edge
                Default=1.0

        Returns:
        Initialize class GraphDmk
        """

        # member with edge number
        self.n_edges = topol.shape[1]

        # member with nodes number
        self.n_nodes = np.amax(topol) + 1 - np.amin(topol)

        # graph topology
        self.topol = cp(topol)

        # edge weight
        if weight is None:
            weight = np.ones(self.n_edges)
        self.weight = weight

    def signed_incidence_matrix(self):
        """
        Build signed incidence matrix

        Args:
        topol: (2,ndege)-integer np-array 1-based ordering

        Result:
        matrix : signed incidence matrix
        """
        # build signed incidence matrix
        indptr = np.zeros([2 * self.n_edges]).astype(int)  # rows
        indices = np.zeros([2 * self.n_edges]).astype(int)  # columns
        data = np.zeros([2 * self.n_edges])  # nonzeros
        for i in range(self.n_edges):
            indptr[2 * i : 2 * i + 2] = int(i)
            indices[2 * i] = int(self.topol[0, i])
            indices[2 * i + 1] = int(self.topol[1, i])
            data[2 * i : 2 * i + 2] = [1.0, -1.0]
            # print(topol[i,:],indptr[2*i:2*i+2],indices[2*i:2*i+2],data[2*i:2*i+2])
        signed_incidence = scipy.sparse.csr_matrix(
            (data, (indptr, indices)), shape=(self.n_edges, self.n_nodes)
        )
        return signed_incidence
