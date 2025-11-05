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

    admk = AdmkSolver(problem, tol_opt=1e-4, tol_constraint=1e-8)
    admk.set_ctrl("verbose", 1)
    admk.set_ctrl("max_iter", 200)
    admk.set_ctrl(["explicit_euler_tdens","deltat","control"], "adaptive")

    admk.solve()
    vel, pot, tdens = admk.get_otp_solution()

    # The potential is defined up to a constant.
    # We can fix the constant by shifting the potential so that the first component is zero.
    pot -= pot[0]

    # The potential is also defined up to a sign flip.
    # We can check both signs and see which one is closer to the expected potential.
    if np.linalg.norm(pot - expected_pot) > np.linalg.norm(-pot - expected_pot):
        pot = -pot

    support = [0,3,4]
        
    diff_pot = np.linalg.norm(pot[support] - expected_pot[support]) / np.linalg.norm(expected_pot[support]) 
    assert diff_pot < 1e-3
    diff_tdens = np.linalg.norm(tdens - expected_tdens) / np.linalg.norm(expected_tdens) 
    assert diff_tdens < 1e-3
