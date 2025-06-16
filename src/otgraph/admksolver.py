import numpy as np
from petsc4py import PETSc
import time as cputiming
import scipy
from copy import deepcopy

from .problems import MinNormProblem
from .petsc_utils import (
    nested_set,
    nested_get,
    flatten_parameters,
    solve_with_petsc,
    scipy2petsc,
)


def adaptive_deltat(state, update):
    order_down = -1
    order_up = 1
    min_u = np.min(update)
    max_u = np.max(update)

    deltat_l = 1e30
    if min_u < 0:
        deltat_l = (10**order_down - 1) / min_u

    deltat_u = 1e30
    if max_u > 0:
        deltat_u = (10**order_up - 1) / max_u

    deltat = min(deltat_l, deltat_u)

    return deltat


class AdmkSolver:
    """
    Solver class for problem
    min \|v\|_{w}^{q} A v = rhs
    with A signed incidence matrix of Graph
    via Algebraic Dynamic Monge-Kantorovich.
    We find the long time solution of the
    dynamics
    \dt \Tdens(t)=\Tdens(t) * | \Grad \Pot(\Tdens)|^2 -Tdens^{gamma}
    """

    def __init__(
        self,
        problem: MinNormProblem,
        tol_opt: float = 1e-3,
        tol_constraint: float = 1e-5,
    ):
        """
        Initialize solver with passed controls (or default)
        and initialize structure to store info on solver application
        """
        self.problem = problem
        self.n_pot = problem.n_row
        self.n_tdens = problem.n_col
        self.sol = self.init_solution()

        # init infos
        self.linear_solver_iterations = 0
        self.nonlinear_solver_iterations = 0
        self.nonlinear_solver_residum = 0.0
        self.n_pot = problem.n_row
        self.n_tdens = problem.n_col

        self.methods_lists = ["explicit_tdens"]
        self.ctrl = self.init_ctrl(
            tol_opt=tol_opt,
            tol_constraint=tol_constraint,
            max_iter=200,
            method="explicit_euler",
        )

        self.deltat = 0.0
        self.time = 0.0

        self.pot_indices = slice(0, self.n_pot)  # np.arange(self.n_pot,dtype='int')
        self.tdens_indices = slice(
            self.n_pot, self.n_pot + self.n_tdens
        )  # np.arange(self.n_tdens,dtype='int')+self.n_pot

        # IS(=Index Set) is the system used by petsc to store indices
        # We use it to assign which dofs are associate to pot and tdens
        self.pot_is = PETSc.IS()
        self.tdens_is = PETSc.IS()
        # self.pot_is.createGeneral(np.arange(self.n_pot,dtype='i'),comm=PETSc.COMM_WORLD)
        # self.tdens_is.createGeneral(np.arange(self.n_tdens,dtype='i')+self.n_pot,comm=PETSc.COMM_WORLD)
        self.pot_is.createStride(
            size=self.n_pot, first=0, step=1, comm=PETSc.COMM_WORLD
        )
        self.tdens_is.createStride(
            size=self.n_tdens, first=self.n_pot, step=1, comm=PETSc.COMM_WORLD
        )

    def get_otp_solution(self):
        """
        Convert dmk solution to the solution of the problem
        """
        pot, tdens = self.subfunctions(self.sol)
        vel = tdens * self.problem.grad.dot(pot)
        return vel, pot, tdens

    ####################################################
    # CONTROLS
    ####################################################
    def init_ctrl(self, tol_opt, tol_constraint, max_iter, method):
        """
        Set the controls of the Dmk solver
        """
        ctrl = {
            "tol_opt": tol_opt,
            "tol_constraint": tol_constraint,
            "max_iter": max_iter,
            "max_restart": 5,
            "method": "explicit_euler_tdens",
            # monitor controls
            "verbose": 0,
            "log": 0,
            "log_file": "admk.log",
            # available methods
            "explicit_euler_tdens": {
                # global controls
                "tdens_min": 1e-8,
                # time stepping controls
                "deltat": {
                    "initial": 0.01,
                    "control": "fixed",
                    "min": 1e-4,
                    "max": 1e0,
                    "expansion": 1.05,
                    "contraction": 2.0,
                },
                # linear solver controls
                "ksp": {
                    "type": "cg",
                    "norm_type": "unpreconditioned",
                },
                "pc": {
                    "type": "hypre",
                    # used if for ilu only
                    "factor_drop_tolerance": {"dt": 1e-4, "maxrowcount": 30},
                },
                # saving of evolution
                "save": {
                    "sol": {
                        "directory": "./runs/",
                        # 'no','some','last'
                        "mode": "no",
                        "frequency": 10,
                    },
                    "matrices": {
                        "directory": "./runs/matrices/",
                        "mode": "no",
                        "frequency": 10,
                    },
                },
            },
            "implicit_euler_gfvar": {
                # global controls
                "tdens_min": 1e-8,
                # time stepping controls
                "deltat": {
                    "initial": 1,
                    "control": "expansive",
                    "min": 1e-4,
                    "max": 1e3,
                    "expansion": 2,
                    "contraction": 2.0,
                },
                # linear solver controls
                "ksp": {
                    "type": "cg",
                    "norm_type": "unpreconditioned",
                },
                "pc": {
                    "type": "hypre",
                    # used if for ilu only
                    "factor_drop_tolerance": {"dt": 1e-4, "maxrowcount": 30},
                },
                "snes": {"type": "nls", "max_it": 20},
            },
            #
            "relax_Laplacian": 1e-10,
            # linear solver controls for first iteration
            "ksp": {
                "type": "cg",
                "rtol": tol_constraint,
                "norm_type": "unpreconditioned",
            },
            "pc": {
                "type": "hypre",
                # used if for ilu only
                "factor_drop_tolerance": {"dt": 1e-4, "maxrowcount": 30},
            },
        }

        return ctrl

    def set_ctrl(self, keys_list, value):
        if not isinstance(keys_list, list):
            keys_list = [keys_list]
        nested_set(self.ctrl, keys_list, value)

    def get_ctrl(self, keys_list):
        if not isinstance(keys_list, list):
            keys_list = [keys_list]
        return nested_get(self.ctrl, keys_list)

    def print_info(self, msg, priority, indent=0):
        """
        Print messagge to stdout and to log
        file according to priority passed
        """
        if self.get_ctrl("verbose") >= priority:
            if indent > 0:
                msg = "   " * indent + msg
            print(msg)

    ####################################################
    # CONTROLS
    ####################################################

    # functions defining how the solver specific variables
    # stored for this algorithm
    def init_solution(self):
        """
        Initialize solution as unique np array = [pot;tdens]
        """
        sol = np.zeros(self.n_pot + self.n_tdens)
        sol[-self.n_tdens :] = 1.0
        return sol

    def subfunctions(self, sol):
        """
        Split solution in pot, tdens component
        """
        pot = sol[: self.n_pot]
        tdens = sol[-self.n_tdens :]
        return pot, tdens

    def solve_pot(self, sol, petsc_options):
        """
        Args:
         sol: np.array [pot,tdens], changed in place
         petsc_options: dictionary for petsc solver

        Returns:
         ierr : control flag (=0 if everthing worked)
        """
        pot, tdens = self.subfunctions(sol)

        # assembly stiff
        msg = f"{min(tdens):.2E}<=TDENS<={max(tdens):.2E}"
        self.print_info(msg, 3, 2)

        # set matrix and rhs
        start_time = cputiming.time()
        diag_tdens = scipy.sparse.diags(tdens)
        stiff = self.problem.div.dot(diag_tdens.dot(self.problem.grad))
        msg = "ASSEMBLY" + "{:.2f}".format(-(start_time - cputiming.time()))
        self.print_info(msg, 3, 2)

        rhs = self.problem.rhs.copy()

        #
        # solve linear system
        #
        relax = self.get_ctrl("relax_Laplacian")
        stiff += relax * scipy.sparse.eye(self.n_pot)  # matrix is singular

        ierr, iters, res, pres = solve_with_petsc(stiff, rhs, pot, petsc_options)

        # info
        msg = f"{ierr=} it={iters:04d}" + f" res={res:.1e}" + f" pres={pres:.1e}"
        self.print_info(msg, priority=2, indent=1)

        return ierr

    def pmass(self):
        return self.problem.q_exponent / (2 - self.problem.q_exponent)

    def mass_function(t, derivative_order=0):
        self.pmass_exponent = self.pmass()
        f = lambda t: t**self.pmass_exponent
        df = f.diff(t)
        dff = df.diff(t)
        t = scipy.symbols("t")
        sympy.lambdafy(t, self.mass_function)
        return

    def g(t):
        pmass_exponent = self.pmass()
        t = spmpy.symbol("t")
        g_fun = t**pmass_exponent / pmass_exponent
        return g_fun

    def lambdafy(g):
        t = g.get
        return sympy.lambdafy(t, g)

    def weight_mass(self, tdens):
        return 0.5 * np.dot(problem.weight * self.mass_function(tdens))

    def Lagrangian(self, pot, tdens):
        """
        Compute Lagrangian of the problem
        """
        grad_pot = self.problem.grad.dot(pot)
        forcing = self.problem.rhs
        L = (
            np.dot(forcing, pot)
            - 0.5 * np.dot(tdens * grad_pot**2, self.problem.weight)
            + 0.5 * np.dot(tdens ** self.pmass(), self.problem.weight)
        )
        return L

    def Lagrangian_gradient(self, pot, tdens, var):
        """
        Compute the gradient w.r.t to variable
        """
        if var == "tdens":
            grad_pot = self.problem.grad.dot(pot)
            pmass = self.problem.q_exponent / (2 - self.problem.q_exponent)
            gradient_tdens = 0.5 * (-(grad_pot**2) + tdens ** (pmass - 1))

            return gradient_tdens

        elif var == "pot":
            grad_pot = self.problem.grad.dot(pot)
            gradient_pot = self.problem.rhs - self.problem.div.dot(tdens * grad_pot)
            # for inode in self.problem.dirichlet_nodes:
            #    gradient_pot[inode] = pot[inode] - self.problem.dirichlet_values[inode]

            return gradient_pot

    def Lagrangian_hessian(self, pot, tdens, var_row, var_col):
        """
        Compute the gradient w.r.t to variable
        """
        if var_row == "tdens":
            if var_col == "tdens":
                pmass = self.problem.q_exponent / (2 - self.problem.q_exponent)
                if abs(pmass - 1.0) < 1e-16:
                    hessian = scipy.sparse.diags(np.zeros(self.n_tdens))
                else:
                    hessian = scipy.sparse.diags(
                        0.5 * tdens ** (pmass - 2) * (pmass - 1)
                    )
                return hessian
            elif var_col == "pot":
                grad_pot = self.problem.grad.dot(pot)
                D = scipy.sparse.diags(-1.0 * grad_pot)
                # matvec = lambda x: grad_pot * self.problem.grad.dot(x)
                # hessian = splinalg.LinearOperator((self.n_pot,self.n_tdens),matvec)
                # sp.sparse.diags(trans_prime * grad_pot).dot(problem.matrixT)
                hessian = D.dot(self.problem.matrixT)
                return hessian

        elif var_row == "pot":
            if var_col == "tdens":
                return self.Lagrangian_hessian(pot, tdens, "tdens", "pot").transpose()
            elif var_col == "pot":
                diag_tdens = scipy.sparse.diags(-1.0 * tdens)
                hessian = self.problem.div.dot(diag_tdens.dot(self.problem.grad))
                return hessian

    def opt_residual(self, sol):
        """
        compute a number to measure if optimality is obtained
        """
        pot, tdens = self.subfunctions(sol)
        gradient_tdens = self.Lagrangian_gradient(pot, tdens, "tdens")
        var = norm(tdens * gradient_tdens * self.problem.weight) / norm(
            tdens * self.problem.weight
        )
        return var

    def tdens2gfvar(self, tdens):
        """
        Transformation from tdens variable to gfvar (gradient flow variable)
        """
        gfvar = np.sqrt(tdens)
        return gfvar

    def gfvar2tdens(self, gfvar, derivative_order=0):
        """
        Compute \phi(gfvar)=tdens, \phi' (gfvar), or \phi''(gfvar)
        """
        if derivative_order == 0:
            tdens = gfvar**2
        elif derivative_order == 1:
            tdens = 2 * gfvar
        elif derivative_order == 2:
            tdens = 2 * np.ones(len(gfvar))
        else:
            print("Derivative order not supported")
        return tdens

    def build_fnewton_gfvar(self, problem, pot, gfvar, gfvar_old, ctrl):
        # assembly nonlinear equation
        # F_pot=f_newton[1:n_pot]= stiff * pot - rhs
        # F_pot=f_newton[1+n_pot:n_pot + n_tdens] = -weight (gfvar-gfvar_old)/deltat + \grad \Lyapunov
        tdens = self.gfvar2tdens(gfvar, 0)  # 1 means first derivative
        trans_prime = self.gfvar2tdens(gfvar, 1)  # 1 means first derivative
        trans_second = self.gfvar2tdens(gfvar, 2)  # 2 means second derivative
        grad_pot = problem.potential_gradient(pot)

        f_newton = np.zeros(npot + ntdens)

        f_newton[0:n_pot] = problem.matrix.dot(tdens * grad_pot) - problem.rhs
        f_newton[n_pot : n_pot + n_tdens] = -problem.weight * (
            (gfvar - gfvar_old) / ctrl.deltat
            + trans_prime * 0.5 * (-(grad_pot**2) + 1.0)
        )
        return -f_newton

    def build_jacobian_gfvar(self, problem, pot, gfvar, ctrl):
        # assembly jacobian
        conductivity = tdens * problem.inv_weight
        A_matrix = self.build_stiff(problem.matrix, conductivity)
        B_matrix = scipy.sparse.diags(trans_prime * grad_pot).dot(problem.matrixT)
        BT_matrix = B_matrix.transpose()

        # the minus sign is to get saddle point in standard form
        diag_C_matrix = problem.weight * (
            1.0 / self.deltat + trans_second * 0.5 * (-(grad_pot**2) + 1.0)
        )
        C_matrix = sp.sparse.diags(diag_C_matrix)
        msg = (
            "{:.2E}".format(min(diag_C_matrix))
            + "<=C <="
            + "{:.2E}".format(max(diag_C_matrix))
        )
        if ctrl.verbose >= 3:
            print(msg)
        return A_matrix, B_matrix, BT_matrix, C_matrix

    def eval_F(self, potgfvar, fnewton):
        pot, gfvar = self.subfunctions(potgfvar)
        tdens = self.gfvar2tdens(gfvar)
        trans_prime = self.gfvar2tdens(gfvar, 1)

        fnewton[self.pot_indices] = self.Lagrangian_gradient(pot, tdens, "pot")
        fnewton[self.tdens_indices] = (
            self.Lagrangian_gradient(pot, tdens, "tdens") * trans_prime
        )
        fnewton[self.tdens_indices] += (
            1 / self.deltat * self.problem.weight * (gfvar - self.gfvar_old)
        )

    def eval_Jacobian(self, potgfvar):
        pot, gfvar = self.subfunctions(potgfvar)
        tdens = self.gfvar2tdens(gfvar)
        trans_prime = self.gfvar2tdens(gfvar, 1)
        trans_second = self.gfvar2tdens(gfvar, 2)

        J11 = self.Lagrangian_hessian(
            pot, tdens, "pot", "pot"
        ) - 1e-10 * scipy.sparse.eye(self.n_pot)
        D_prime = scipy.sparse.diags(trans_prime)
        J12 = self.Lagrangian_hessian(pot, tdens, "pot", "tdens").dot(D_prime)
        J21 = J12.transpose()
        J22 = (
            D_prime.dot(self.Lagrangian_hessian(pot, tdens, "tdens", "tdens"))
            + scipy.sparse.diags(
                trans_second * self.Lagrangian_gradient(pot, tdens, "tdens")
            )
            + scipy.sparse.diags(1 / self.deltat * self.problem.weight)
        )

        return J11, J12, J21, J22

    def F_block(self, snes, potgfvar_petsc, F_petsc):
        potgfvar = potgfvar_petsc.array

        pot, gfvar = self.subfunctions(potgfvar)
        tdens = self.gfvar2tdens(gfvar)
        trans_prime = self.gfvar2tdens(gfvar, 1)

        fnewton = np.zeros(self.n_pot + self.n_tdens)
        fnewton[self.pot_indices] = self.Lagrangian_gradient(pot, tdens, "pot")
        fnewton[self.tdens_indices] = (
            self.Lagrangian_gradient(pot, tdens, "tdens") * trans_prime
        )
        fnewton[self.tdens_indices] += (
            1 / self.deltat * self.problem.weight * (gfvar - self.gfvar_old)
        )

        F_petsc.setArray(fnewton)

    def J_block(self, snes, potgfvar_petsc, J_petsc, P_petsc):
        potgfvar = potgfvar_petsc.array
        J11, J12, J21, J22 = self.eval_Jacobian(potgfvar)
        petsc_J11 = scipy2petsc(J11)
        petsc_J12 = scipy2petsc(J12)
        petsc_J21 = scipy2petsc(J21)
        petsc_J22 = scipy2petsc(J22)

        J_petsc = PETSc.Mat().createNest(
            [[petsc_J11, petsc_J12], [petsc_J21, petsc_J22]]
        )
        P_petsc = J_petsc

    def iterate(self, sol):
        """
        Procedure overriding update of parent class(Problem)

        Args:
        problem: Class with inputs  (rhs, q_exponent)
        sol  : Class with unkowns (tdens, pot in this case)

        Returns:
         sol : update sol from time t^k to t^{k+1}

        """
        method = self.get_ctrl("method")
        method_ctrl = self.get_ctrl(method)

        if method == "explicit_euler_tdens":
            # check that residual is below threesold
            pot, tdens = self.subfunctions(sol)
            gradient_pot = self.Lagrangian_gradient(pot, tdens, "pot")
            res_pot = np.linalg.norm(gradient_pot) / np.linalg.norm(self.problem.rhs)
            if res_pot > self.get_ctrl("tol_constraint"):
                petsc_options = flatten_parameters(
                    {
                        # get main linear solver controls
                        **{"ksp": nested_get(method_ctrl, ["ksp"])},
                        **flatten_parameters({"pc": nested_get(method_ctrl, ["pc"])}),
                        **{
                            # we start from previous solution
                            "ksp_initial_guess_nonzero": True,
                            # we solve the linear system to reach the tolerance
                            # of the constraint since
                            # div(v)-f = div(tdens grad pot) -f
                            "ksp_rtol": self.get_ctrl("tol_constraint"),
                            #'ksp_monitor_true_residualot': None,
                        },
                    }
                )

                ierr = self.solve_pot(sol, petsc_options)

            # compute update direction
            gradient_tdens = self.Lagrangian_gradient(pot, tdens, "tdens")
            update = -tdens * gradient_tdens

            # set time step deltat
            deltat_ctrl = nested_get(method_ctrl, ["deltat", "control"])

            if deltat_ctrl == "fixed":
                pass
            if deltat_ctrl == "expansive":
                self.deltat *= nested_get(method_ctrl, ["deltat", "expansion"])
            if deltat_ctrl == "adaptive":
                self.deltat = adaptive_deltat(tdens, update)

            self.deltat = max(self.deltat, nested_get(method_ctrl, ["deltat", "min"]))
            self.deltat = min(self.deltat, nested_get(method_ctrl, ["deltat", "max"]))

            msg = bounds(tdens, "TDENS")
            self.print_info(msg, 2, 1)

            msg = bounds(update, "UPDATE") + f" deltat={self.deltat:.2e}"
            self.print_info(msg, 2, 1)

            # update tdens
            tdens += self.deltat * update
            tdens_min = nested_get(method_ctrl, ["tdens_min"])
            tdens[np.where(tdens < tdens_min)] = tdens_min
            # no need to set sol since tdens is just a pointer

            # set linear solvers
            petsc_options = flatten_parameters(
                {
                    # get main linear solver controls
                    **{"ksp": nested_get(method_ctrl, ["ksp"])},
                    **flatten_parameters({"pc": nested_get(method_ctrl, ["pc"])}),
                    **{
                        # we start from previous solution
                        "ksp_initial_guess_nonzero": True,
                        # we solve the linear system to reach the tolerance
                        # of the constraint since
                        # div(v)-f = div(tdens grad pot) -f
                        "ksp_rtol": self.get_ctrl("tol_constraint"),
                        #'ksp_monitor_true_residualot': None,
                    },
                }
            )
            ierr = self.solve_pot(sol, petsc_options)

            if ierr == 0:
                self.time += self.deltat

            return ierr

        elif method == "implicit_euler_gfvar":
            self.deltat = 1.0

            fnewton = np.zeros(self.n_pot + self.n_tdens)

            it = 0
            inc = np.zeros(self.n_pot + self.n_tdens)
            x = np.zeros(self.n_pot + self.n_tdens)
            ierr = 0

            pot, tdens = self.subfunctions(sol)
            gfvar = self.tdens2gfvar(tdens)

            x[self.pot_indices] = pot[:]
            x[self.tdens_indices] = gfvar[:]

            self.gfvar_old = np.zeros(self.n_tdens)
            self.gfvar_old[:] = gfvar[:]

            # self.pc.setType('fieldsplit')
            ksp_options = {"ksp_type": "preonly", "pc_type": "lu"}
            ksp_options = {
                "ksp_type": "fgmres",
                "ksp_rtol": 1e-10,
                "ksp_monitor_true_residual": None,
                #
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",  # based on schur complement
                "pc_fieldsplit_schur_fact_type": "full",  # use full factorization
                # A B^T = (I         )(A  )(I A^{-1})
                # B  -C   (BA^{-1} I )(  S)(  I     )
                # TODO : swap order of fields, now is not working and we need to swap
                # when pc_setfieldsplit
                "pc_fieldsplit_0_fields": "1,",  # field 1
                "pc_fieldsplit_1_fields": "0,",  # field 0
                "pc_fieldsplit_schur_precondition": "selfp",  # form Sp=A+B^T C^{-1} B                               }
                "fieldsplit_0": {
                    "ksp_type": "preonly",
                    "pc_type": "jacobi",
                },
                "fieldsplit_1": {
                    "ksp_type": "preonly",
                    "pc_type": "hypre",
                },
                "info": None,
            }
            ksp_options = flatten_parameters(ksp_options)

            use_snes = False
            if use_snes:
                snes_options = {"snes": {"type": "nls", "rtol": 1e-6, "monitor": None}}

                petsc_options = {
                    **flatten_parameters(ksp_options),
                    **flatten_parameters(snes_options),
                }
                print(petsc_options)

                problem_prefix = "nonlinear_solver_"

                # Setup SNES solver
                snes = PETSc.SNES().create()

                opts = PETSc.Options()
                opts.prefixPush(problem_prefix)
                for k, v in petsc_options.items():
                    opts[k] = v
                    opts.prefixPop()

                pc = snes.ksp.getPC()
                pc.setFromOptions()
                pc.setFieldSplitIS(("0", self.pot_is), ("1", self.tdens_is))
                pc.setOptionsPrefix(problem_prefix)
                pc.setFromOptions()

                # this is allocation
                J11, J12, J21, J22 = self.eval_Jacobian(x)
                petsc_J11 = scipy2petsc(J11)
                petsc_J12 = scipy2petsc(J12)
                petsc_J21 = scipy2petsc(J21)
                petsc_J22 = scipy2petsc(J22)

                petsc_J = PETSc.Mat().createNest(
                    [[petsc_J11, petsc_J12], [petsc_J21, petsc_J22]]
                )

                petsc_F = petsc_J.createVecLeft()
                petsc_x = petsc_J.createVecRight()
                petsc_x.setArray(x)

                # assign funcition, Jacobian, and Jacobian for preconditioner
                snes.setFunction(self.F_block, petsc_F)
                snes.setJacobian(self.J_block, petsc_J, P=None)

                snes.setFromOptions()
                snes.solve(None, petsc_x)
                x[:] = petsc_x.getArray()
                pot, gfvar = self.subfunctions(x)
                snes_converged = snes.getConvergedReason()
                print(snes_converged)
                self.eval_F(potgfvar, fnewton)
                fnorm = np.linalg.norm(fnewton)
                print(f"|F|{fnorm:.1e}")
                if ierr_newton == 0:
                    sol[self.pot_indices] = pot[:]
                    sol[self.tdens_indices] = self.gfvar2tdens(gfvar)[:]

                return 0

            it = 0
            max_iter = 20
            ierr_newton = 0
            self.deltat = 1.0
            while it <= max_iter:
                self.eval_F(x, fnewton)
                fnewton *= -1.0
                fnorm = np.linalg.norm(fnewton)
                print(f"{it=:03d} |F|{fnorm:.1e}")
                if fnorm < 1e-6:
                    break

                if it >= max_iter:
                    ierr_newton = 2
                    break

                J11, J12, J21, J22 = self.eval_Jacobian(x)
                # J = bmat([[J11,J12],[J21,J22]],format='csr')
                # J = self.eval_Jacobian(x)

                # petsc_J = scipy2petsc(J)
                petsc_J11 = scipy2petsc(J11)
                petsc_J12 = scipy2petsc(J12)
                petsc_J21 = scipy2petsc(J21)
                petsc_J22 = scipy2petsc(J22)

                petsc_J = PETSc.Mat().createNest(
                    [[petsc_J11, petsc_J12], [petsc_J21, petsc_J22]]
                )

                petsc_inc = petsc_J.createVecLeft()
                petsc_rhs = petsc_J.createVecRight()

                L_sizes = petsc_J.getSizes()
                L_range = petsc_J.getOwnershipRange()
                print("L_sizes", L_sizes)
                neqns = L_sizes[0][0]
                print("neqns", neqns)

                problem_prefix = "jacobian_solver_"

                # copy from https://github.com/FEniCS/dolfinx/blob/230e027269c0b872c6649f053e734ed7f49b6962/python/dolfinx/fem/petsc.py#L618
                # https://github.com/FEniCS/dolfinx/fem/petsc.py
                opts = PETSc.Options()
                opts.prefixPush(problem_prefix)
                for k, v in ksp_options.items():
                    opts[k] = v
                    print(k, v)
                opts.prefixPop()
                opts.view()

                ksp = PETSc.KSP().create()
                ksp.setOperators(petsc_J)
                ksp.setOptionsPrefix(problem_prefix)
                # assign fieldsplit
                pc = ksp.getPC()
                pc.setFromOptions()

                pc.setFieldSplitIS(("0", self.tdens_is), ("1", self.pot_is))

                # pc.setFieldSplitFields(self.n_tdens,('0','1'))
                # pc.setFieldSplitFields(self.n_pot,('1','0'))
                # pc.setFieldSplitIS(('0',self.pot_is),('1',self.tdens_is))
                pc.setOptionsPrefix(problem_prefix)
                pc.setFromOptions()

                ksp.setConvergenceHistory()
                ksp.setUp()
                ksp.setFromOptions()
                pc.view()
                petsc_J.setOptionsPrefix(problem_prefix)
                petsc_J.setFromOptions()

                petsc_inc.setOptionsPrefix(problem_prefix)
                petsc_inc.setFromOptions()

                petsc_rhs.setOptionsPrefix(problem_prefix)
                petsc_rhs.setFromOptions()

                # convert to petsc
                petsc_rhs.setArray(fnewton)
                petsc_inc.setArray(inc)

                # solve
                ksp.solve(petsc_rhs, petsc_inc)

                reason = ksp.getConvergedReason()
                last_pres = ksp.getResidualNorm()
                if reason < 0:
                    ierr_newton = 1
                rhs_norm = petsc_rhs.norm()

                last_iter = ksp.getIterationNumber()
                h = ksp.getConvergenceHistory()
                if len(h) > 0:
                    resvec = h[-(last_iter + 1) :]

                    res = 0
                    if rhs_norm > 0:
                        res = resvec[-1] / rhs_norm

                    last_pres = ksp.getResidualNorm()
                    pres = last_pres

                # convert back to np
                inc[:] = petsc_inc.getArray()

                # res = norm(J.dot(inc)-fnewton)

                x += inc

                pot, gfvar = self.subfunctions(x)
                it += 1

                print(
                    f"{it=} {ierr_newton=}| linear solver {res=:.1e} iter={last_iter}"
                )
            print(ksp_options)
            ksp.view()

            if ierr_newton == 0:
                sol[self.pot_indices] = pot[:]
                sol[self.tdens_indices] = self.gfvar2tdens(gfvar)[:]
            else:
                ierr = ierr_newton

            return ierr

    def solve(self, initial_solution=None, after_update_callback=None):
        """
        Solve the time dependent problem
        Args:
            problem: problem object
            sol: solution object
            ctrl: control object
        Returns:
            ierr: error code (0: success)
        """
        # use stored solution
        sol = self.sol
        if initial_solution is not None:
            sol[:] = initial_solution[:]

        # solve first Laplacian
        petsc_options = flatten_parameters(
            {
                **{"ksp": self.get_ctrl("ksp")},
                **{"ksp_rtol": self.get_ctrl("tol_constraint")},
                **{"pc": self.get_ctrl("pc")},
            }
        )
        ierr = self.solve_pot(sol, petsc_options)

        # Start main cycle
        iter = 0
        while (ierr == 0) and (iter < self.get_ctrl("max_iter")):
            # try to update the solution
            sol_old = deepcopy(sol)
            nrestart = 0
            ierr_iterate = 0
            while ierr_iterate == 0:

                ierr_iterate = self.iterate(sol)
                if ierr_iterate == 0:
                    break
                else:
                    sol = deepcopy(sol_old)
                    nrestart += 1
                    if nrestart == self.get_ctrl("max_restart"):
                        break

                    method = self.get_ctrl("method")
                    if method == "explicit_euler":
                        self.deltat /= 2.0

            # check if the iteration has been successful
            if ierr_iterate != 0:
                ierr = 1
                self.ierr_update = ierr_iterate

            print(f"{ierr=}", iter, self.ctrl.get("max_iter"))

            # check if the maximum number of iterations has been reached
            iter += 1
            if iter == self.ctrl.get("max_iter"):
                ierr = 2

            # Here the user evalutes if convergence is achieved
            pot, tdens = self.subfunctions(self.sol)
            gradient_pot = self.Lagrangian_gradient(pot, tdens, "pot")
            res = np.linalg.norm(gradient_pot) / np.linalg.norm(self.problem.rhs)

            gradient_tdens = self.Lagrangian_gradient(pot, tdens, "tdens")
            var = np.linalg.norm(
                tdens * gradient_tdens * self.problem.weight
            ) / np.linalg.norm(tdens * self.problem.weight)
            msg = f"it={iter} var={var:.2e} res{res:.2e}"
            self.print_info(msg, 1)
            if (var < self.get_ctrl("tol_opt")) and (
                res < self.get_ctrl("tol_constraint")
            ):
                ierr = 0
                print("DONE")
                break

            if after_update_callback is not None:
                after_update_callback(self, sol)

        return ierr

    def ierr_reason(self, ierr):
        """
        Return a description of the error
        """
        if ierr == 0:
            return "No error"
        if ierr == 1:
            return "Error in iterate procedure"
        if ierr == 2:
            return "Maximum number of iterations reached"
