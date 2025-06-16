import numpy as np
from otgraph import Graph, MinNormProblem
from otgraph import AdmkSolver


def test_main(verbose=0):
    # 0---1
    # | X |
    # 2---3---4
    # set the topology of the graph
    topol = np.array([[0, 1], [0, 3], [2, 3], [1, 2], [0, 2], [1, 3], [3, 4]])
    # set the weight of the weight
    weight = np.ones(7)

    # aasign initial and final configuration of Optimal Transport
    source = np.zeros(5)
    target = np.zeros(5)
    source[0] = 1
    target[1:] = 0.25

    # this must sum to zero
    rhs = source - target

    # optpot is the distance from the node-0
    optimal_pot = np.zeros(5)
    optimal_pot[0] = 0.0
    optimal_pot[1:4] = 1.0
    optimal_pot[4] = 2.0

    # Init. graph problem and matrix
    graph = Graph(topol.T)

    # Init. signed incidence matrix
    incidence_matrix = graph.signed_incidence_matrix()
    E_matrix = incidence_matrix.transpose()

    # Init problem inputs (rhs, q, exponent)
    problem = MinNormProblem(E_matrix, rhs, q_exponent=1.0, weight=weight)

    # Init solver
    admk = AdmkSolver(problem, tol_opt=1e-3, tol_constraint=1e-8)

    admk.set_ctrl(
        ["explicit_euler_tdens", "deltat"],
        {
            "control": "adaptive",
            "initial": 1e-1,
            "min": 1e-2,
            "max": 5e-1,
            "expansion": 1.05,
            "contraction": 2.0,
        },
    )
    admk.set_ctrl("method", "implicit_euler_gfvar")
    # linear solver
    # matrix is singualr. we need to relax it with + relax*identity
    admk.set_ctrl("relax_Laplacian", 1e-10)
    admk.set_ctrl(["ksp", "type"], "preonly")
    admk.set_ctrl(["pc", "type"], "lu")
    admk.set_ctrl(["pc", "factor_drop_tolerance", "dt"], 1e-4)

    admk.set_ctrl("verbose", 2)
    admk.set_ctrl("max_iter", 50)

    # run solver
    # get mu, pot, vel
    admk.solve()
    vel, u, mu = admk.get_otp_solution()

    # shift potential to get zero at the root node
    u -= u[0]
    u *= -1.0

    # print results
    if verbose > 0:
        print("The potential u in this case is minus the distance from the first node")
        print(f"{u=}")
        print("The transport density mu count the mass passing throught each edge")
        print(f"{mu=}")
        print("The velocity=mu gradient (pot) describes the flux of mass")
        print(f"{vel=}")

    print(u)
    print(optimal_pot)
    print(np.linalg.norm(u - optimal_pot))

    assert np.allclose(u, optimal_pot, rtol=1e-3)


if __name__ == "__main__":
    test_main()
