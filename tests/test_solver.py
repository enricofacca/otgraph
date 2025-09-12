import numpy as np
from otgraph import Graph, MinNormProblem
from otgraph import AdmkSolver


def test_solver_case_1():
    # Test case based on the user's request
    # graph with topology [[0,1],[1,2],[2,3],[3,0],[3,4]]
    # source = [1,0,0,0,0]
    # sink=[0,0,0,0,1]
    # solution pot=[0,1,2,1,2] (up to constant or flip of the sign)
    # tdens=[0,0,0,1,1]

    topol = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [3, 4]])
    source = np.array([1, 0, 0, 0, 0])
    sink = np.array([0, 0, 0, 0, 1])
    expected_pot = np.array([0, 1, 2, 1, 2])
    expected_tdens = np.array([0, 0, 0, 1, 1])

    rhs = source - sink

    graph = Graph(topol.T)
    incidence_matrix = graph.signed_incidence_matrix()
    E_matrix = incidence_matrix.transpose()

    problem = MinNormProblem(E_matrix, rhs, q_exponent=1.0)

    admk = AdmkSolver(problem, tol_opt=1e-3, tol_constraint=1e-8)
    admk.set_ctrl("verbose", 0)
    admk.set_ctrl("max_iter", 50)

    admk.solve()
    vel, pot, tdens = admk.get_otp_solution()

    # The potential is defined up to a constant.
    # We can fix the constant by shifting the potential so that the first component is zero.
    pot -= pot[0]

    # The potential is also defined up to a sign flip.
    # We can check both signs and see which one is closer to the expected potential.
    if np.linalg.norm(pot - expected_pot) > np.linalg.norm(-pot - expected_pot):
        pot = -pot

    assert np.allclose(pot, expected_pot, rtol=1e-3, atol=1e-3)
    assert np.allclose(tdens, expected_tdens, rtol=1e-3, atol=1e-3)
