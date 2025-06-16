import numpy as np
from petsc4py import PETSc


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


def scipy2petsc(J):
    print(J.shape)
    print(type(J))
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
