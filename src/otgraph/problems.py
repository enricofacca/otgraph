import numpy as np
import scipy.sparse.linalg as splinalg


class MinNormProblem:
    """
    This class contains the inputs of problem GraphDmk
    min |v|^{q>=1}_w : A v = rhs
    with
    - |v|^{q>=1}_w = sum_{i} |v_i|^q* w_i
      where w is a strictly positive vector
    - A signed incidence matrix of graph G
      rows number = number of nodes
      columns number = number of edges
    - rhs_of_time = right-hand side. it can be a function of time
    - q_exponent = exponent of the norm
    - weight = weight in the norm
    """

    def __init__(self, matrix, rhs, q_exponent=1.0, weight=None):
        """
        Constructor of problem setup
        """
        self.matrix = matrix
        self.n_row = matrix.shape[0]
        self.n_col = matrix.shape[1]

        self.matrixT = self.matrix.transpose()

        # edge weight
        if weight is None:
            weight = np.ones(self.n_col)
        self.weight = weight
        self.inv_weight = 1.0 / weight

        # gradient W^{-1} * (signed_incidence * potential)
        def matvec_grad(x):
            return self.inv_weight * self.matrixT.dot(x)
        self.gradient = splinalg.LinearOperator((self.n_col, self.n_row), matvec_grad)

        self.rhs = rhs
        self.q_exponent = q_exponent

        ierr = self.check_inputs()
        if ierr != 0:
            print("Error in inputs")

        self.grad = self.gradient
        self.div = self.matrix

    def check_inputs(self):
        """
        Method to check problem inputs consistency
        """
        ierr = 0
        balance = np.sum(self.rhs) / np.linalg.norm(self.rhs)
        if balance > 1e-11:
            print(f"Rhs is not balanced {balance:.1E}")
            ierr = 1
        return ierr

    def potential_gradient(self, pot):
        """
        Procedure to compute gradient of the potential
        grad=W^{-1} A^T pot

        Args:
        pot: real (nrow of A)-np.array with potential

        Returns:
        grad: real (ncol of A)-np.array with gradient
        """
        grad = self.inv_weight * self.matrixT.dot(pot)
        return grad

    def constraint_residual(self, vel):
        """
        Procedure to compute residual of the constraint
        res = A vel - rhs

        Args:
        vel: real (ncol of A)-np.array with velocity

        Returns:
        res: real (nrow of A)-np.array with residual
        """
        rhs_norm = np.linalg.norm(self.rhs)
        res = np.linalg.norm(self.div.dot(vel) - self.rhs) / rhs_norm
        return res
