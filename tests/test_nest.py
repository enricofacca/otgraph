from petsc4py import PETSc
from otgraph.petsc_utils import diag, setup_ksp_solver
import numpy as np



# def diag(diag_vector,type="diagonal"):
#     """
#     Return the diagonal matrix with the entries of diag_vec
#     """
#     if type == "diagonal":
#         D = PETSc.Mat().createDiagonal(diag_vector)
#     if type == "aij":
#         D = PETSc.Mat().createAIJ(size=diag_vector.getSize())
#         D.setUp()
#         D.setDiagonal(diag_vector)
#     return D

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
        #print("PrimalPC")
        #print(appctx)
        self.vpp = appctx["vpp"]
        #print("PrimalPC vpp", self.vpp.getArray())
        
        self.vpt = appctx["vpt"]
        #print("PrimalPC vpt", self.vpt.getArray())
        
        self.vts = appctx["vts"]
        #print("PrimalPC vts", self.vts.getArray())
        
        self.vst = appctx["vst"]
        #print("PrimalPC vst", self.vst.getArray())
        
        self.vss = appctx["vss"]
        #print("PrimalPC vss", self.vss.getArray())
        
        self.c =  self.vts * self.vst / self.vss
        self.inv_v = 1.0 / (self.vpp + self.vpt * self.vpt / self.c)
        #print("PrimalPC inv_v", self.inv_v.getArray())
        self.inv_S = diag(self.inv_v)

    def apply(self, pc, x, y):
        #print("PrimalPC apply")
        #print("x", x.getArray())
        #self.inv_v.pointwiseMult(x,y)
        self.inv_S.mult(x, y)
        #print("y", y.getArray())

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
        self.vst = appctx["vst"]
        self.vts = appctx["vts"]
        self.vss = appctx["vss"]
        # c = - vst/ vss 
        self.c = self.vts * self.vst / self.vss
        self.inv_c = -1.0 / self.c
        #print("CPC inv_c", self.inv_c.getArray())
        #self.x_copy = PETSc.Vec().createWithArray(np.zeros(self.v21.getSize()))
        
        self.inv_C = diag(self.inv_c)


    def apply(self, pc, x, y):
        print("CPC apply")
        #print("x", x.getArray())
        #x.copy(self.x_copy)
        #print("x_copy", self.x_copy.getArray())
        #print("inv_c", self.inv_c.getArray())
        self.inv_C.mult(x, y)
        #print("y", y.getArray())






# Create submatrices
vpp = PETSc.Vec().createWithArray([10, 10])
App = diag(vpp)

vpt = PETSc.Vec().createWithArray([1, 2])
Apt = diag(vpt)
Atp = diag(vpt)



vts = PETSc.Vec().createWithArray([2, 3])
Ats = diag(vts)

# A12 = PETSc.Mat().create()
# A12.setSizes([2, 2])
# A12.setFromOptions()
# A12.setUp()
# A12.setValues([0, 1], [0, 1], [5.0, 6.0, 7.0, 8.0])
# A12.assemblyBegin()
# A12.assemblyEnd()
vst = PETSc.Vec().createWithArray([3, 2])
Ast = diag(vst)


vss = PETSc.Vec().createWithArray([3, 3])
Ass = diag(vss)

vc = PETSc.Vec().createWithArray(-vst/vss)
C = diag(vc)

# Create nested matrix
#A = PETSc.Mat().createNest([[A00, A01, None], [A10, None, A12], [None,A21,A22]])
A = PETSc.Mat().createNest([[Ass, Ast, None], [Ats, None, Atp], [None,Apt,App]])
A.setUp()

# Create nested matrix
#A = PETSc.Mat().createNest([[A00, A01, None], [A10, None, A12], [None,A21,A22]])
#A = PETSc.Mat().createNest([[C, A10], [A01,A00]])
#A.setUp()

iss_rows, iss_cols = A.getNestISs()
print("ISs in the nested matrix:")
for i, is_ in enumerate(iss_rows):
    print(f"IS {i}: {is_.getIndices()}")

for i, is_ in enumerate(iss_cols):
    print(f"IS {i}: {is_.getIndices()}")
# Set the values for the nested matrix

# Print the nested matrix structure
#A.view()

#A_dense = PETSc.Mat().createDense([6, 6])
#A.convert(PETSc.Mat.Type.DENSE, A_dense)


# Print the dense matrix
#A_dense.view()


# 
# set ksp controls
#
ksp_ctrl = {
    "ksp" : {
        "type":"preonly",
        "max_it" : 100,
        "rtol" : 1e-6,
        "monitor_true_residual" : None,
    },
    "pc_type" : "fieldsplit",
    "pc_fieldsplit_type" : "schur",
    "pc_fieldsplit_schur_fact_type" : "full",
    # how to split the fields
    "pc_fieldsplit_block_size" : 3,
    "pc_fieldsplit_0_fields": "0", # slack
    "pc_fieldsplit_1_fields": "1, 2", # tdens, pot
    # tdens ^{-1}
    "fieldsplit_0" : {
        "ksp_type": "preonly",
        "pc_type": "jacobi"
        },
    # (2x2) inverse
    "fieldsplit_1" : {
        "ksp_type": "preonly",
        "pc_type" : "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "pc_fieldsplit_block_size" : 2,
        "pc_fieldsplit_0_fields": "0", # tdens
        "pc_fieldsplit_1_fields": "1", # pot
        # C^{-1}
        "fieldsplit_0" : {
            "ksp_type" : "preonly",
            "pc_type": "python",
            "pc_python_type": __name__+ ".CPC",
        },
        # primal schur with cond = tdens + tdens/slack g^2
        "fieldsplit_1" : {
            "ksp_type" : "preonly",
            "pc_type": "python",
            "pc_python_type": __name__ + ".PrimalPC",
        }
    }
}



sol_vec = PETSc.Vec().createWithArray(np.zeros(6))
rhs_vec = PETSc.Vec().createWithArray(np.zeros(6))
rhs_vec.set(1.0)

appctx={"vpp": vpp, "vpt" : vpt, "vst":vst, "vts":vts,"vss":vss}

solver_prefix = "iterative_solver_"
ksp = setup_ksp_solver(
        A,
        solver_options=ksp_ctrl,
        field_ises= [
            ("0", iss_rows[0]), 
            ("1", iss_rows[1]), 
            ("2", iss_rows[2])
            ],
        appctx=appctx,
        solver_prefix= solver_prefix
    )

#sol_vec.setOptionsPrefix(solver_prefix)
#sol_vec.setFromOptions()

#rhs_vec.setOptionsPrefix(solver_prefix)
#rhs_vec.setFromOptions()


ksp.solve(rhs_vec, sol_vec)

res = A.createVecRight()
A.mult(sol_vec, res)
res.axpy(-1.0, rhs_vec)

print("Residual vector after initial solve:")
assert(res.norm()/rhs_vec.norm() < 1e-10)


print("Solution vector after iterative solver:")
print(sol_vec.getArray())


sol_direct_vec = PETSc.Vec().createWithArray(np.zeros(6))
rhs_vec.set(1.0)


ksp_ctrl = {
    "ksp_type":"preonly",
    "pc_type":"lu"
}

ksp_direct = setup_ksp_solver(
        A,
        solver_options=ksp_ctrl,
        solver_prefix= "direct_solver_"
    )

ksp_direct.solve(rhs_vec, sol_direct_vec)
print("Solution vector after direct solver:")
print(sol_direct_vec.getArray())


A.mult(sol_vec, res)
res.axpy(-1.0, rhs_vec)

print("Residual vector after initial solve:")
assert( res.norm()/rhs_vec.norm() < 1e-10 )

assert( (sol_vec-sol_direct_vec).norm() < 1e-10 )

A = PETSc.Mat().createNest([ 
    [App, Apt, None], 
    [Atp, None, Ats], 
    [None, Ast, Ass]])
A.setUp()

# Create nested matrix
#A = PETSc.Mat().createNest([[A00, A01, None], [A10, None, A12], [None,A21,A22]])
#A = PETSc.Mat().createNest([[C, A10], [A01,A00]])
#A.setUp()

iss_rows, iss_cols = A.getNestISs()


# 
# set ksp controls
#
ksp_ctrl = {
    "ksp" : {
        "type":"preonly",
        "max_it" : 100,
        "rtol" : 1e-6,
        "monitor_true_residual" : None,
    },
    "pc_type" : "fieldsplit",
    "pc_fieldsplit_type" : "schur",
    "pc_fieldsplit_schur_fact_type" : "full",
    # how to split the fields
    "pc_fieldsplit_block_size" : 3,
    "pc_fieldsplit_0_fields": "2", # slack
    "pc_fieldsplit_1_fields": "0, 1", # tdens, pot
    # tdens ^{-1}
    "fieldsplit_0" : {
        "ksp_type": "preonly",
        "pc_type": "jacobi"
        },
    # (2x2) inverse
    "fieldsplit_1" : {
        "ksp_type": "preonly",
        "pc_type" : "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_fact_type": "full",
        "pc_fieldsplit_block_size" : 2,
        "pc_fieldsplit_0_fields": "1", # tdens
        "pc_fieldsplit_1_fields": "0", # pot
        # C^{-1}
        "fieldsplit_0" : {
            "ksp_type" : "preonly",
            "pc_type": "python",
            "pc_python_type": __name__+ ".CPC",
        },
        # primal schur with cond = tdens + tdens/slack g^2
        "fieldsplit_1" : {
            "ksp_type" : "preonly",
            "pc_type": "python",
            "pc_python_type": __name__ + ".PrimalPC",
        }
    }
}


solver_prefix = "other_solver_"
ksp = setup_ksp_solver(
        A,
        solver_options=ksp_ctrl,
        field_ises= [
            ("0", iss_rows[0]), 
            ("1", iss_rows[1]), 
            ("2", iss_rows[2])
            ],
        appctx=appctx,
        solver_prefix= solver_prefix
    )


sol_vec.set(1.0)
ksp.solve(rhs_vec, sol_vec)
print("Solution vector after other solver:")
print(sol_vec.getArray())

res = A.createVecRight()
A.mult(sol_vec, res)
res.axpy(-1.0, rhs_vec)

print("Residual vector after initial solve:")
assert(res.norm()/rhs_vec.norm() < 1e-10)


