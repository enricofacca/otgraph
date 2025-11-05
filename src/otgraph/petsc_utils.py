import numpy as np
from petsc4py import PETSc
from typing import Optional, Union
import numpy.typing as npt




def _make_reasons(reasons):
    return dict(
        [(getattr(reasons, r), r) for r in dir(reasons) if not r.startswith("_")]
    )


SNESReasons = _make_reasons(PETSc.SNES.ConvergedReason())
KSPReasons = _make_reasons(PETSc.KSP.ConvergedReason())


#
# following code is taken from firedrake
#
def flatten_parameters(parameters, sep="_"):
    """Flatten a nested parameters dict, joining keys with sep.

    :arg parameters: a dict to flatten.
    :arg sep: separator of keys.

    Used to flatten parameter dictionaries with nested structure to a
    flat dict suitable to pass to PETSc.  For example:

    .. code-block:: python3

       flatten_parameters({"a": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}

    If a "prefix" key already ends with the provided separator, then
    it is not used to concatenate the keys.  Hence:

    .. code-block:: python3

       flatten_parameters({"a_": {"b": {"c": 4}, "d": 2}, "e": 1}, sep="_")
       => {"a_b_c": 4, "a_d": 2, "e": 1}
       # rather than
       => {"a__b_c": 4, "a__d": 2, "e": 1}
    """
    new = type(parameters)()

    if not len(parameters):
        return new

    def flatten(parameters, *prefixes):
        """Iterate over nested dicts, yielding (*keys, value) pairs."""
        sentinel = object()
        try:
            option = sentinel
            for option, value in parameters.items():
                # Recurse into values to flatten any dicts.
                for pair in flatten(value, option, *prefixes):
                    yield pair
            # Make sure zero-length dicts come back.
            if option is sentinel:
                yield (prefixes, parameters)
        except AttributeError:
            # Non dict values are just returned.
            yield (prefixes, parameters)

    def munge(keys):
        """Ensure that each intermediate key in keys ends in sep.

        Also, reverse the list."""
        for key in reversed(keys[1:]):
            if len(key) and not key.endswith(sep):
                yield key + sep
            else:
                yield key
        else:
            yield keys[0]

    for keys, value in flatten(parameters):
        option = "".join(map(str, munge(keys)))
        if option in new:
            print(
                "Ignoring duplicate option: %s (existing value %s, new value %s)",
                option,
                new[option],
                value,
            )
        new[option] = value
    return new


def nested_set(dic, keys, value, create_missing=True):
    d = dic
    for key in keys[:-1]:
        if key in d:
            d = d[key]
        elif create_missing:
            d = d.setdefault(key, {})
        else:
            return dic
    if keys[-1] in d or create_missing:
        d[keys[-1]] = value
    return dic


def nested_get(dic, keys):
    d = dic
    for key in keys[:-1]:
        if key in d:
            d = d[key]

    value = d[keys[-1]]
    return value

def diag(diag_vector,type="diagonal"):
    """
    Return the diagonal matrix with the entries of diag_vec
    """
    if type == "diagonal":
        D = PETSc.Mat().createDiagonal(diag_vector)
    if type == "aij":
        D = PETSc.Mat().createAIJ(size=diag_vector.getSize())
        D.setUp()
        D.setDiagonal(diag_vector)
    return D

def scipy2petsc(J):
    tmp = J
    if J.getformat() != "csr":
        tmp = J.tocsr()

    petsc_J = PETSc.Mat().createAIJ(
        size=tmp.shape, csr=(tmp.indptr, tmp.indices, tmp.data)
    )
    return petsc_J


def solve_with_petsc(stiff, rhs: np.array, pot: np.array, petsc_options: dict):
    """
    Solve linear system with petsc
    """

    petsc_stiff = PETSc.Mat().createAIJ(
        size=stiff.shape, csr=(stiff.indptr, stiff.indices, stiff.data)
    )

    petsc_pot = petsc_stiff.createVecLeft()
    petsc_rhs = petsc_stiff.createVecRight()

    problem_prefix = "laplacian_solver_"
    ksp = PETSc.KSP().create()
    ksp.setOperators(petsc_stiff)
    ksp.setOptionsPrefix(problem_prefix)

    # copy from https://github.com/FEniCS/dolfinx/blob/230e027269c0b872c6649f053e734ed7f49b6962/python/dolfinx/fem/petsc.py#L618
    # https://github.com/FEniCS/dolfinx/fem/petsc.py
    opts = PETSc.Options()
    opts.prefixPush(problem_prefix)
    for k, v in petsc_options.items():
        opts[k] = v
    opts.prefixPop()
    ksp.setConvergenceHistory()
    # ksp.pc.setReusePreconditioner(True) # not sure if this is needed
    ksp.setFromOptions()
    petsc_stiff.setOptionsPrefix(problem_prefix)
    petsc_stiff.setFromOptions()
    petsc_pot.setOptionsPrefix(problem_prefix)

    petsc_rhs.setFromOptions()

    ierr = 0
    iter = 0
    res = 0
    pres = 0

    # convert to petsc
    petsc_rhs.setArray(rhs)
    petsc_pot.setArray(pot)

    # solve
    ksp.solve(petsc_rhs, petsc_pot)

    # store info
    reason = ksp.getConvergedReason()
    last_pres = ksp.getResidualNorm()
    if reason < 0:
        ierr = 1
        return ierr
    else:
        last_iter = ksp.getIterationNumber()
        iter += last_iter
        h = ksp.getConvergenceHistory()
        if len(h) > 0:
            resvec = h[-(last_iter + 1) :]
            rhs_norm = petsc_rhs.norm()
            if rhs_norm > 0:
                res = max(res, resvec[-1] / rhs_norm)

        last_pres = ksp.getResidualNorm()
        pres = max(pres, last_pres)

        if res > petsc_options["ksp_rtol"]:
            print(
                f"{KSPReasons[reason]=} {res=} rhs={petsc_rhs.norm()} pot={petsc_pot.norm()}"
            )

    # get solution
    pot[:] = petsc_pot.getArray()

    return ierr, iter, res, pres


def bounds(vect, label: str):
    return f"{np.min(vect):.1e}<={label}<={np.max(vect):.1e}"


def setup_ksp_solver(
        A_petsc: PETSc.Mat,
        solver_options: dict={},
        field_ises: Optional[
            list[tuple[str, Union[PETSc.IS, npt.NDArray[np.int32]]]]
        ] = None,
        nullspace: Optional[list[np.ndarray]] = None,
        appctx: dict = None,
        solver_prefix: str = "petsc_solver_",
    ) -> PETSc.KSP:
        """
        KSP solver for PETSc matrices

        Parameters
        ----------
        A : sps.csc_matrix
            Matrix to solve
        field_ises : Optional[list[tuple[str,PETSc.IS]]], optional
            Fields index sets, by default None.
            This tells how to partition the matrix in blocks for field split
            (block-based) preconditioners.

            Example with IS:
            is_0 = PETSc.IS().createStride(size=3, first=0, step=1)
            is_1 = PETSc.IS().createStride(size=3, first=3, step=1)
            [('0',is_0),('1',is_1)]
            Example with numpy array:
            [('flux',np.array([0,1,2],np.int32)),('pressure',np.array([3,4,5],np.int32))]
        nullspace : Optional[list[np.ndarray]], optional
            Nullspace vectors, by default None
        appctx : dict, optional
            Application context, by default None.
            It is attached to the KSP object to gather information that can be used
            to form the preconditioner.
        solver_prefix : str, optional
        """


        # Convert kernel np array. to petsc nullspace
        # TODO: ensure orthogonality of the nullspace vectors
        # convert to petsc vectors
        comm = PETSc.COMM_WORLD
        petsc_kernels = []
        if nullspace is not None:
            for v in nullspace:
                p_vec = PETSc.Vec().createWithArray(v, comm=comm)
                petsc_kernels.append(p_vec)
            petsc_nullspace = PETSc.NullSpace().create(
                constant=None, vectors=petsc_kernels, comm=comm
            )
            A_petsc.setNullSpace(petsc_nullspace)
        else:
            petsc_nullspace = None


        
        # set field_ises
        if field_ises is not None:
            if isinstance(field_ises[0][1], PETSc.IS):
                field_ises = field_ises
            if isinstance(field_ises[0][1], np.ndarray):
                field_ises = [
                    (i, PETSc.IS().createGeneral(is_i)) for i, is_i in field_ises
                ]
        else:
            field_ises = None

        
        
    # def setup(self, petsc_options: dict, ) -> None:
        petsc_options = flatten_parameters(solver_options)

        # create ksp solver and assign controls
        ksp = PETSc.KSP().create()
        ksp.setOperators(A_petsc)
        ksp.setOptionsPrefix(solver_prefix)


        ksp.setConvergenceHistory()
        ksp.setFromOptions()
        opts = PETSc.Options()
        opts.prefixPush(solver_prefix)
        for k, v in petsc_options.items():
            opts[k] = v
            #print(f"Setting option {k} = {v}")
        opts.prefixPop()
        ksp.setConvergenceHistory()
        ksp.setFromOptions()

        
        # associate petsc vectors to prefix
        A_petsc.setOptionsPrefix(solver_prefix)
        A_petsc.setFromOptions()

        #sol_vec.setOptionsPrefix(solver_prefix)
        #sol_vec.setFromOptions()

        #rhs_vec.setOptionsPrefix(solver_prefix)
        #rhs_vec.setFromOptions()

        #ksp.view()

        # TODO: set field split in __init__
        if (field_ises) is not None and (
            "fieldsplit" in petsc_options["pc_type"]
        ):
            pc = ksp.getPC()
            pc.setFromOptions()
            # syntax is
            # pc.setFieldSplitIS(('0',is_0),('1',is_1))
            pc.setFieldSplitIS(*field_ises)
            pc.setOptionsPrefix(solver_prefix)
            pc.setFromOptions()
            pc.setUp()
            #ksp.view()

            # split subproblems
            ksps = pc.getFieldSplitSubKSP()
            for i,k in enumerate(ksps):
                # Without this, the preconditioner is not set up
                # It works for now, but it is not clear why
                #k.setUp()
                p = k.getPC()
                
                # This is in order to pass the appctx
                # to all preconditioner. TODO: find a better way
                p.setAttr("appctx", appctx)
                p.setUp()
                #ksp.view()

                if "fieldsplit" in p.getType():
                    kspksp = p.getFieldSplitSubKSP()
                    for kk in kspksp:
                        # Without this, the preconditioner is not set up
                        # It works for now, but it is not clear why
                        pp = kk.getPC()

                    
                        # This is in order to pass the appctx
                        # to all preconditioner. TODO: find a better way
                        pp.setAttr("appctx", appctx)
                        pp.setUp()

            # set nullspace
            if petsc_nullspace is not None:
                if len(petsc_kernels) > 1:
                    raise NotImplementedError(
                        "Nullspace currently works with one kernel only"
                    )
                # assign nullspace to each subproblem
                for i, local_ksp in enumerate(ksps):
                    for k in petsc_kernels:
                        sub_vec = k.getSubVector(field_ises[i][1])
                        if sub_vec.norm() > 0:
                            local_nullspace = PETSc.NullSpace().create(
                                constant=False, vectors=[sub_vec]
                            )
                            A_i, _ = local_ksp.getOperators()
                            A_i.setNullSpace(local_nullspace)

        ksp.setFromOptions()
        # attach info to ksp
        ksp.setAttr("appctx", appctx)

        return ksp





class ImplicitProduct(object):
    """
    Given two operators retunrns its implicit product
    """
    def __init__(self,op1,op2):
        self.op1 = op1
        self.op2 = op2
        self._tmp = op1.createVecLeft()
        self._tmp2 = op2.createVecRight()

    def mult(self,mat,x,y):
        self.op2.mult(x,self._tmp)
        self.op1.mult(self._tmp,y)
    
    def multTranspose(self,mat,x,y):
        self.op1.multTranspose(x,self._tmp2)
        self.op2.multTranspose(self._tmp2,y)

    def as_matrix(self):
        """
        Return the Schur complement as a matrix
        """
        A = PETSc.Mat().create()
        A.setSizes([self.op1.size[0],self.op1.size[1]])
        A.setType(A.Type.PYTHON)
        A.setPythonContext(self)
        A.setUp()
        return A
