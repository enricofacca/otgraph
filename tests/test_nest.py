from petsc4py import PETSc
from otgraph.petsc_utils import diag, setup_ksp_solver
import numpy as np


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
        print("PrimalPC apply")
        x_c = x.copy()
        print("x pot", x_c.getArray())
        #self.inv_v.pointwiseMult(x,y)
        self.inv_S.mult(x, y)
        print("y pot", y.getArray())

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

    def apply(self, pc, x, y):
        print("invC apply")
        x_c = x.copy()
        print("x tdens", x_c.getArray())
        y.pointwiseMult(self.inv_c,x)
        print("y tdens", y.getArray())
        
class SSPC(object):
    """
    This is a test for building my own preconditioner,
    getting the extra info from the dictionary appctx passed
    to the linear solver. 
    We are trying to replate what is done in firedrake.
    """
    def setUp(self,pc):
        # get info from the parent KSP object
        appctx = pc.getAttr("appctx")
        self.vss = appctx["vss"]
        self.inv_vss = 1.0 / self.vss

    def apply(self, pc, x, y):
        print("invSS apply")
        x_c = x.copy()
        print("x slack", x_c.getArray())
        y.pointwiseMult(self.inv_vss,x)
        print("y slack", y.getArray())


# Create submatrices
vpp = PETSc.Vec().createWithArray([10, 10])
App = diag(vpp)

vpt = PETSc.Vec().createWithArray([1, 2])
Apt = diag(vpt)
Atp = diag(vpt)

vts = PETSc.Vec().createWithArray([2, 3])
Ats = diag(vts)

vst = PETSc.Vec().createWithArray([3, 2])
Ast = diag(vst)

vss = PETSc.Vec().createWithArray([3, 3])
Ass = diag(vss)

vc = PETSc.Vec().createWithArray(-vst/vss)
C = diag(vc)

appctx={"vpp": vpp, "vpt" : vpt, "vst":vst, "vts":vts,"vss":vss}

def test_slacktdenspot(appctx):
    App = diag(appctx["vpp"])
    Apt = diag(appctx["vpt"])
    Atp = diag(appctx["vpt"])
    Ats = diag(appctx["vts"])
    Ast = diag(appctx["vst"])
    Ass = diag(appctx["vss"])

    
    # Create nested matrix
    A = PETSc.Mat().createNest(
        [[Ass, Ast,  None],
         [Ats, None, Atp],
         [None,Apt,  App]])
    A.setUp()


    rhs_vec = PETSc.Vec().createWithArray(np.zeros(6))
    rhs_vec.set(1.0)


    sol_direct_vec = PETSc.Vec().createWithArray(np.zeros(6))
    ksp_ctrl = {
        "ksp_type":"preonly",
        "pc_type":"lu"
    }

    ksp_direct = setup_ksp_solver(
        A,
        rhs_vec,
        sol_direct_vec,
        solver_options=ksp_ctrl,
        solver_prefix= "direct_solver_"
    )

    ksp_direct.solve(rhs_vec, sol_direct_vec)
    print("Solution vector after direct solver:")
    print(sol_direct_vec.getArray())



        
    # 
    # set ksp controls
    #
    ksp_ctrl = {
        "ksp" : {
            "type": "preonly",
            "max_it" : 10,
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
            "pc_type": "python",
            "pc_python_type": __name__+ ".SSPC",
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

    # Create nested matrix
    iss_rows, iss_cols = A.getNestISs()
    print("ISs in the nested matrix:")
    for i, is_ in enumerate(iss_rows):
        print(f"IS {i}: {is_.getIndices()}")

    for i, is_ in enumerate(iss_cols):
        print(f"IS {i}: {is_.getIndices()}")




    sol_vec = PETSc.Vec().createWithArray(np.zeros(6))
    solver_prefix = "iterative_solver_"
    ksp = setup_ksp_solver(
        A,
        rhs_vec,
        sol_vec,
        solver_options=ksp_ctrl,
        field_ises= [
            ("0", iss_rows[0]), 
            ("1", iss_rows[1]), 
            ("2", iss_rows[2])
            ],
        appctx=appctx,
        solver_prefix= solver_prefix
    )
    ksp.solve(rhs_vec, sol_vec)
    res = A.createVecRight()
    A.mult(sol_vec, res)
    res.axpy(-1.0, rhs_vec)

    print("Solution vector after iterative solver:")
    print(sol_vec.getArray())


    
    print("Residual vector after initial solve:")
    res_norm = res.norm()/rhs_vec.norm()
    print(f"{res_norm:.2e}")
    assert res_norm < 1e-12
    
    print("Direct vs iterative")
    diff_norm = (sol_vec-sol_direct_vec).norm()
    print(f"{diff_norm=:.2e}")
    assert( diff_norm < 1e-10 )


def test_pottdensslack(appctx):
    App = diag(appctx["vpp"])
    Apt = diag(appctx["vpt"])
    Atp = diag(appctx["vpt"])
    Ats = diag(appctx["vts"])
    Ast = diag(appctx["vst"])
    Ass = diag(appctx["vss"])




    # Create nested matrix
    A = PETSc.Mat().createNest([
        [App,  Apt,  None],
        [Atp,  None, Ats ],
        [None, Ast,  Ass ],
    ])
    A.setUp()

    
    rhs_vec = PETSc.Vec().createWithArray(np.zeros(6))
    rhs_vec.set(1.0)

    sol_direct_vec = PETSc.Vec().createWithArray(np.zeros(6))
    ksp_ctrl = {
        "ksp_type":"preonly",
        "pc_type":"lu"
    }
    ksp_direct = setup_ksp_solver(
        A,
        rhs_vec,
        sol_direct_vec,
        solver_options=ksp_ctrl,
        solver_prefix= "direct_solver_"
    )

    ksp_direct.solve(rhs_vec, sol_direct_vec)
    print("Solution vector after direct solver:")
    print(sol_direct_vec.getArray())




    
    # Create nested matrix
    iss_rows, iss_cols = A.getNestISs()
    solver_prefix = "iterative_solver_swapped2"

    sol_vec = PETSc.Vec().createWithArray(np.zeros(6))

    # 
    # set ksp controls
    #
    ksp_ctrl2 = {
        "ksp" : {
        "type": "preonly",
        "max_it" : 10,
        "rtol" : 1e-6,
        "monitor_true_residual" : None,
    },
    "pc_type" : "fieldsplit",
    "pc_fieldsplit_type" : "schur",
    "pc_fieldsplit_schur_fact_type" : "full",
    # how to split the fields
    "pc_fieldsplit_block_size" : 3,
    "pc_fieldsplit_0_fields": "2",    # slack
    "pc_fieldsplit_1_fields": "0, 1", # pot, tdens
    # tdens ^{-1}
    "fieldsplit_0" : {
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": __name__+ ".SSPC",
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


    ksp = setup_ksp_solver(
        A,
        rhs_vec,
        sol_vec,
        solver_options=ksp_ctrl2,
        field_ises= [
            ("0", iss_rows[0]), 
            ("1", iss_rows[1]), 
            ("2", iss_rows[2])
            ],
        appctx=appctx,
        solver_prefix= solver_prefix
    )
    ksp.solve(rhs_vec, sol_vec)
    res = A.createVecRight()
    A.mult(sol_vec, res)
    res.axpy(-1.0, rhs_vec)

    print("Solution vector after iterative solver:")
    print(sol_vec.getArray())

    
    print("Residual vector after initial solve:")
    res_norm = res.norm()/rhs_vec.norm()
    print(f"{res_norm:.2e}")
    assert res_norm < 1e-12

    print("Compare Direct and iterative")
    assert( (sol_vec-sol_direct_vec).norm() < 1e-10 )

print()
print("slacktdenspot")
print()
test_slacktdenspot(appctx)
print()
print("pottdensslack")
print()
test_pottdensslack(appctx)
