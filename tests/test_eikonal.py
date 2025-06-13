import numpy as np
from otgraph import Graph, MinNormProblem


def test_main(verbose=0):
    # set the topology of the graph
    topol = np.array([[1, 2], [1, 4], [3, 4], [2, 3], [1, 3], [2, 4], [4, 5]])
    # 0-index
    topol -= 1

    # set the weight of the weight
    weight = np.ones(7)

    # aasign initial and final configuration of Optimal Transport
    source = np.zeros(5)
    target = np.zeros(5)
    source[0] = 1
    target[1:] = 0.25

    # this must sum to zero
    rhs = source - target

    # Init. graph problem and matrix
    graph = Graph(topol.T)

    # Init. signed incidence matrix
    incidence_matrix = graph.signed_incidence_matrix()
    E_matrix = incidence_matrix.transpose()

    # Init problem inputs (rhs, q, exponent)
    problem = MinNormProblem(E_matrix, rhs, q_exponent=1.0, weight=weight)

    assert np.allclose(problem.rhs, rhs, rtol=1e-10)
