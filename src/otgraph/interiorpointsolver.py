"""
Solve using interior point methods.
"""
import numpy as np
from petsc4py import PETSc
from .petsc_utils import setup_ksp_solver


class ImplicitJacobian(object):
    """
    Given two operators retunrns its implicit product
    """
    def __init__(self, solver,  sol):
        self.solver = solver
        self.sol = sol
        self.pot = sol.getSubVector(solver.pot_is)
        self.tdens = sol.getSubVector(solver.tdens_is)
        self.slack = sol.getSubVector(solver.slack_is)
        
        self.grad_pot = solver.problem.grad.createVecLeft()
        solver.problem.grad.Mult(self.pot, self.grad_pot)

        self._tmp_m = solver.problem.matrix.createVecRight()
        self._tmp_m_bis = solver.problem.matrix.createVecRight()
        self._tmp_m = solver.problem.matrix.createVecLeft()



    def mult(self,mat,x,y):
        x_pot = x.getSubVector(self.solver.pot_is)
        x_tdens = x.getSubVector(self.solver.tdens_is)
        x_slack = x.getSubVector(self.solver.slack_is)

        y.set(0.0)
        # stiff x_pot
        self.problem.grad.Mult(x_pot, self._tmp_m)
        self._tmp_m.PointwiseMult(x_tdens, self._tmp_m_bis)
        self.solver.problem.matrixA.mult(self._tmp_m_bis, self._tmp_n)
        y.isaxpy(self.solver.pot_is, 1.0, self._tmp_n)

        # A DG x_tdens
        self.grad_pot.PointwiseMult(x_tdens, self._tmp_m)
        self.solver.problem.matrixA.mult(self._tmp_m, self._tmp_n)
        y.isaxpy(self.solver.pot_is, 1.0, self._tmp_n)

        # 

        # Dg A^T x_pot
        self.problem.matrixA.multTranspose(x_pot, self._tmp_m)
        self._tmp_m_bis.PointwiseMult(self.grad_pot, self._tmp_m)
        y.isaxpy(self.solver.tdens_is, 1.0, self._tmp_m)
        # S x_slack
        y.isaxpy(self.solver.tdens_is, 1.0, x_slack)

        #
        #
        #
        self.slack.PointwiseMult(x_tdens, self._tmp_m)
        y.isaxpy(self.solver.slack_is, 1.0, self._tmp_m)
        self.tdens.PointwiseMult(x_slack, self._tmp_m)
        y.isaxpy(self.solver.slack_is, 1.0, self._tmp_m)
        
    
    def as_petsc_matrix(self):
        """
        Return the Schur complement as a matrix
        """
        A = PETSc.Mat().create()
        A.setSizes([self.op1.size[0],self.op1.size[1]])
        A.setType(A.Type.PYTHON)
        A.setPythonContext(self)

        # set kernel
        v = np.zeros(self.sol.size)
        v[0:self.solver.npot] = 1.0
        v /= np.linalg.norm(v)
        v_vec = PETSc.Vec().createWithArray(v, comm=PETSc.COMM_WORLD)
        nullspace = PETSc.NullSpace().create(constant=None, vectors=[v_vec], comm=PETSc.COMM_WORLD)    
        A.setNullSpace(nullspace)
        A.setUp()
        return A

class InteriorPointSolver:
    """
    A class to solve optimization problems using interior point methods.
    """

    def __init__(self, problem):
        """
        Initialize the InteriorPointSolver.
        """
        
        # Define sizes and create solution vector
        self.npot = problem.npot
        self.ntdens = problem.ntdens
        self.nslack = problem.ntdens
        self.nsol = self.npot + self.nslack + self.ntdens
        
        self.sol_vec = PETSc.Vec().create(size=self.nsol)

        self.pot_is = PETSc.IS().createStride(
            self.npot, 0, 1, PETSc.DECIDE
        )
        self.tdens_is = PETSc.IS().createStrided(
            self.ntdens, self.npot, 1, PETSc.DECIDE
        )
        self.slack_is = PETSc.IS().createStrided(
            self.nslack, self.npot + self.ntdens, 1, PETSc.DECIDE
        )
        self.eps = 1.0

        """ Create vectors for the solution, gradient, and residuals."""
        self.ntdens_temp_vec = PETSc.Vec().create(size=self.ntdens)
        self.ntdens_temp2_vec = PETSc.Vec().create(size=self.ntdens)

        pass


    
        


    def nonlinear_residual(self, sol_vec: PETSc.Vec, eps: float, residual_vec: PETSc.Vec):
        """
        Compute the nonlinear residual for the optimization problem.

        :param x: The solution vector.
        :return: The nonlinear residual vector.
        """
        pot_vec = sol_vec[self.pot_is]
        tdens_vec = sol_vec[self.tdens_is]
        slack_vec = sol_vec[self.slack_is]
        

        residual_vec.set(0.0)
        f = residual_vec.getSubVector(self.pot_is)
        g = residual_vec.getSubVector(self.tdens_is)
        h = residual_vec.getSubVector(self.slack_is)

        # f = A tdens W ^{-1} grad pot - forcing_pot
        self.grad.Mult(pot_vec, self.ntdens_temp_vec)
        tdens_vec.PointwiseMult(self.ntdens_temp_vec, self.ntdens_temp2_vec)
        self.matrixA.multT(self.ntdens_temp2_vec,f)
        f -= self.problem.forcing_pot

        # g = 0.5 (|grad pot|^2 - 1) + slack
        self.ntdens_temp_vec.pointwiseMult(self.ntdens_temp_vec, g)
        g -= 1.0
        g.scale(0.5)
        g += slack_vec
        
        # h = tdens slack - eps
        tdens_vec.pointwiseMult(slack_vec, h)
        h -= eps

   
    def build_stiff(self, cond_vec: PETSc.Vec):
        """
        Solve for the potential using the current solution vector.

        :param sol_vec: The solution vector.
        :param eps: The epsilon value for the optimization problem.
        :return: The updated solution vector.
        """

        # assemble stiffness matrix and forcing vector
        diag_tdens = PETSc.Mat().createDiag(cond_vec)
        div_tdens = self.problem.div.MatMult(diag_tdens)
        stiff = div_tdens.MatMult(self.problem.grad)
        
        return stiff
    
    def newton(self, sol_vec: PETSc.Vec, eps: float, tolerance: float = 1e-6):
        """
        Perform Newton.

        :param sol_vec: The solution vector.
        :param eps: The epsilon value for the optimization problem.
        :return: The updated solution vector.
        """
        for iteration in range(self.ctrl["newton_max_iter"]):
            # Compute the nonlinear residual
            self.nonlinear_residual(sol_vec, eps, self.residual_vec)

            # Check convergence
            if self.residual_vec.norm() < tolerance:
                print("Converged.")
                return
            

            # Build the Jacobian matrix
            jacobian = ImplicitJacobian(self, sol_vec)

            # 
            # set ksp controls
            #
            ksp_ctrl = {
                "ksp_type" : "gmres",
                "ksp_max_it" : 100,
                "ksp_rtol" : 1e-6,
                "pc_type" : "fielssplit",
                "pc_fieldsplit_type" : "schur",
                "pc_fieldsplit_schur_fact_type" : "full",
                # how to split the fields
                "pc_fieldsplit_block_size" : 3,
                "pc_fieldsplit_0_fields": "0,1", # pot, tdens
                "pc_fieldsplit_1_fields": "2", # slack
                # tdens ^{-1}
                "fieldsplit_1" : {
                    "ksp_type":"preonly",
                    "pc_type":"jacobi"
                    },
                # (2x2) inverse
                "fieldsplit_0" : {
                    "ksp_type":"preonly",
                    "pc_type" : "fieldsplit",
                    "pc_fieldsplit_type": "schur",
                    "pc_fieldsplit_schur_fact_type": "full",
                    "fieldsplit_0_fields": "0", # pot
                    "fieldsplit_1_fields": "1", # tdens
                    # C^{-1}
                    "fieldsplit_1" : {
                        "ksp_type" : "preonly",
                        "pc_type": "python",
                        "pc_python_type": __name__+ "Cblock",
                    },
                    # primal schur with cond = tdens + tdens/slack g^2
                    "fieldsplit_0" : {
                        "ksp_type" : "preonly",
                        "pc_type": "python",
                        "pc_python_type": __name__ + "Primal",
                    }
                }
            }               
                


            # Solve the linear system
            jacobian_solver = setup_ksp_solver(
                    jacobian,
                    self.residual_vec,
                    self.increment_vec,
                    solver_options=ksp_ctrl,
                    field_ises= [
                        ("0", self.pot_is), 
                        ("1", self.tdens_is), 
                        ("2", self.slack_is)],
                    appctx={"solver" : self},
                    solver_prefix= "petsc_solver_")
            
            jacobian_solver.solve()

            # Update the solution vector
            sol_vec.axpy(-1.0, self.increment_vec)


    def build_jacobian(self, sol_vec: PETSc.Vec):
        """
        Build the Jacobian matrix for the optimization problem.

        :param sol_vec: The solution vector.
        :return: The Jacobian matrix.
        """
        pot = sol_vec.getSubVector(self.pot_is)
        tdens = sol_vec.getSubVector(self.tdens_is)
        slack = sol_vec.getSubVector(self.slack_is)


        # create a petsc matrix A * diag(grad_pot)
        self.stiff = self.build_stiff(tdens)
        
        # create the gradient of the potential
        self.problem.grad.mult(pot, self.grad_pot_vec)
        self.DG = PETSc.Mat().createDiag(self.grad_pot_vec)
        self.ADG = self.problem.matrixA.matMult(self.DG)
        self.DGAT = self.ADG.copy()
        self.DGAT.transpose()

        # 
        Id = PETSc.Mat().createConstantDiagonal(self.ntdens,1.0)
        Dtdens = PETSc.Mat().createDiag(tdens)
        Dslack = PETSc.Mat().createDiag(slack)

        # create using createnestmatrix
        jacobian = PETSc.Mat().createNest(
            [
                [self.stiff, self.ADG , None],
                [self.DGAT, None, Id],
                [None, Dslack, Dtdens]
            ],
            comm=PETSc.COMM_WORLD
        )
        jacobian.setUp()

        return jacobian    

    def solve(self):
        """
        Solve the given optimization problem using interior point methods.

        :param problem: The optimization problem to solve.
        :return: The solution to the problem.
        """

        for iteration in range(self.ctrl["max_iter"]):
            # reduce eps
            self.eps *= self.ctrl["eps_reduction_factor"]

            # Solve with newton
            self.newton(self.sol_vec, self.eps, 1e-6)


        

        
    def get_otp_solution(self, format="petsc"):
        """
        Get the solution of the optimization problem.

        :return: The solution vector.
        """
        # Extract the potential, density, and slack variables from the solution vector
        pot = self.sol_vec.getSubVector(self.pot_is)
        tdens = self.sol_vec.getSubVector(self.tdens_is)
        self.problem.grad.Mult(pot, self.ntdens_temp_vec)
        self.ntdens_temp_vec.pointwiseMult(tdens, self.ntdens_temp2_vec)
        
        if format == "petsc":
            return self.ntdens_temp2_vec,  pot, tdens
        else:
            raise ValueError(f"Unknown format: {format}. Supported formats: 'petsc'.")
        

class PrimalPC(object):
    """
    This is a test for building my own preconditioner,
    getting the extra info from the dictionary appctx passed
    to the linear solver. 
    We are trying to replate what is done in firedrake.
    """
    def setUp(self,pc):
        # get info from the parent KSP object
        appctx = pc.getAttr("appctx")
        solver = appctx["solver"]
        sol = appctx["sol"]
        grad_pot = appctx["grad_pot"]

        tdens = sol.getSubVector(solver.tdens_is)
        slack = sol.getSubVector(solver.slack_is)
        
        cond = tdens + tdens / slack * grad_pot**2
        S = solver.build_stiff(cond)
        
        self.rhs = PETSc.create(solver.npot)
        self.sol = PETSc.create(solver.npot)
        self.solver = create(self.S,
                             self.rhs,
                             self.sol,
                             solver_options={"ksp_type":"preonly","pc_type":"hypre"}
                             )

    def apply(self, pc, x, y):
        x.copy(self.rhs)
        self.solver.solve(self.rhs,self.sol)
        self.sol.copy(y)

class CPC(object):
    """
    This is a test for building my own preconditioner,
    getting the extra info from the dictionary appctx passed
    to the linear solver. 
    We are trying to replate what is done in firedrake.
    """
    def setUp(self,pc):
        # get info from the parent KSP object
        appctx = pc.getAttr("appctx")
        solver = appctx["solver"]
        sol = appctx["sol"]
        tdens = sol.getSubVector(solver.tdens_is)
        slack = sol.getSubVector(solver.slack_is)
        
        self.invC = tdens / slack

    def apply(self, pc, x, y):
        self.invC.piontwisemultiply(x, y)




