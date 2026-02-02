import sys
import numpy as np
from petsc4py import PETSc

class PoissonNeumannStag:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.dx = 1.0 / nx
        self.dy = 1.0 / ny

        self.dm = PETSc.DMStag().create(dim=2)
        self.dm.setGlobalSizes([nx, ny])
        self.dm.setDof((0, 1, 1))
        self.dm.setStencilWidth(1)
        self.dm.setStencilType(PETSc.DMStag.StencilType.BOX)
        self.dm.setBoundaryTypes([PETSc.DM.BoundaryType.GHOSTED, PETSc.DM.BoundaryType.GHOSTED])
        self.dm.setUp()

        self.dm.setCoordinateDMType(PETSc.DM.Type.STAG)
        self.dm.setUniformCoordinates(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        self.slot_down = self.dm.getLocationSlot(PETSc.DMStag.StencilLocation.DOWN, 0)
        self.slot_left = self.dm.getLocationSlot(PETSc.DMStag.StencilLocation.LEFT, 0)
        self.slot_elem = self.dm.getLocationSlot(PETSc.DMStag.StencilLocation.ELEMENT, 0)

        (self.gxs, self.gys), (self.gxm, self.gym) = self.dm.getGhostCorners()
        self.lgmap_indices = self.dm.getLGMap().getIndices()
        self.entries_per_elem = self.dm.getEntriesPerElement()

        vec_template = self.dm.createGlobalVector()
        self.n_global = vec_template.getSize()
        self.n_local = vec_template.getLocalSize()
        vec_template.destroy()

    def create_aij_matrix(self):
        # Ensure local size matches DMStag distribution
        M = PETSc.Mat().createAIJ(size=[(self.n_local, self.n_global), (self.n_local, self.n_global)])
        M.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        M.setUp()
        return M

    def get_global_index(self, i, j, slot):
        i_loc = i - self.gxs
        j_loc = j - self.gys
        idx_loc = (j_loc * self.gxm + i_loc) * self.entries_per_elem + slot
        if 0 <= idx_loc < self.lgmap_indices.size:
            return self.lgmap_indices[idx_loc]
        else:
            return -1

    def set_values(self, vec, func_elem=None, func_flux_x=None, func_flux_y=None):
        (start_indices, widths, extra) = self.dm.getCorners()
        xs, ys = start_indices[0], start_indices[1]
        xm, ym = widths[0], widths[1]
        xe, ye = xs + xm, ys + ym
        ex, ey = extra[0], extra[1]

        indices = []
        values = []

        if func_elem:
            for j in range(ys, ye):
                for i in range(xs, xe):
                    x = (i + 0.5) * self.dx
                    y = (j + 0.5) * self.dy
                    idx = self.get_global_index(i, j, self.slot_elem)
                    if idx >= 0:
                        indices.append(idx)
                        values.append(func_elem(x, y))

        if func_flux_x:
            for j in range(ys, ye):
                for i in range(xs, xe + ex):
                    x = i * self.dx
                    y = (j + 0.5) * self.dy
                    idx = self.get_global_index(i, j, self.slot_left)
                    if idx >= 0:
                        indices.append(idx)
                        values.append(func_flux_x(x, y))

        if func_flux_y:
            for j in range(ys, ye + ey):
                for i in range(xs, xe):
                    x = (i + 0.5) * self.dx
                    y = j * self.dy
                    idx = self.get_global_index(i, j, self.slot_down)
                    if idx >= 0:
                        indices.append(idx)
                        values.append(func_flux_y(x, y))

        vec.setValues(indices, values)
        vec.assemble()

    def build_operators(self):
        G = self.create_aij_matrix()
        D = self.create_aij_matrix()

        (start_indices, widths, extra) = self.dm.getCorners()
        xs, ys = start_indices[0], start_indices[1]
        xm, ym = widths[0], widths[1]
        xe, ye = xs + xm, ys + ym
        ex, ey = extra[0], extra[1]
        print(f"{self.dm.comm.rank=} {xs=} {ys=}")
        print(f"{self.dm.comm.rank=} {xm=} {ym=}")
        print(f"{self.dm.comm.rank=} {xe=} {ye=}")
        print(f"{self.dm.comm.rank=} {ex=} {ey=}")
        
        # G: Faces (rows) -> Elements (cols)
        # Left Faces
        for j in range(ys, ye):
            for i in range(xs, xe + ex):
                row_idx = self.get_global_index(i, j, self.slot_left)
                if row_idx < 0: continue

                is_left_boundary = (i == 0)
                is_right_boundary = (i == self.nx)

                if not is_left_boundary and not is_right_boundary:
                    col_idx_center = self.get_global_index(i, j, self.slot_elem)
                    col_idx_west = self.get_global_index(i-1, j, self.slot_elem)
                    G.setValues([row_idx], [col_idx_center, col_idx_west], [1.0/self.dx, -1.0/self.dx], PETSc.InsertMode.ADD_VALUES)

        # Down Faces
        for j in range(ys, ye + ey):
            for i in range(xs, xe):
                row_idx = self.get_global_index(i, j, self.slot_down)
                if row_idx < 0: continue

                is_down_boundary = (j == 0)
                is_up_boundary = (j == self.ny)

                if not is_down_boundary and not is_up_boundary:
                    col_idx_center = self.get_global_index(i, j, self.slot_elem)
                    col_idx_south = self.get_global_index(i, j-1, self.slot_elem)
                    G.setValues([row_idx], [col_idx_center, col_idx_south], [1.0/self.dy, -1.0/self.dy], PETSc.InsertMode.ADD_VALUES)

        # D: Elements (rows) -> Faces (cols)
        for j in range(ys, ye):
            for i in range(xs, xe):
                row_idx = self.get_global_index(i, j, self.slot_elem)
                if row_idx < 0: continue

                col_left = self.get_global_index(i, j, self.slot_left)
                D.setValues([row_idx], [col_left], [-1.0/self.dx], PETSc.InsertMode.ADD_VALUES)

                if i < self.nx:
                    col_right = self.get_global_index(i+1, j, self.slot_left)
                    D.setValues([row_idx], [col_right], [1.0/self.dx], PETSc.InsertMode.ADD_VALUES)

                col_down = self.get_global_index(i, j, self.slot_down)
                D.setValues([row_idx], [col_down], [-1.0/self.dy], PETSc.InsertMode.ADD_VALUES)

                if j < self.ny:
                    col_up = self.get_global_index(i, j+1, self.slot_down)
                    D.setValues([row_idx], [col_up], [1.0/self.dy], PETSc.InsertMode.ADD_VALUES)

        G.assemble()
        D.assemble()
        return G, D

    def assemble_laplacian(self):
        A = self.create_aij_matrix()

        (start_indices, widths, extra) = self.dm.getCorners()
        xs, ys = start_indices[0], start_indices[1]
        xm, ym = widths[0], widths[1]
        xe, ye = xs + xm, ys + ym
        ex, ey = extra[0], extra[1]

        # 1. Element Equations
        for j in range(ys, ye):
            for i in range(xs, xe):
                row_idx = self.get_global_index(i, j, self.slot_elem)
                if row_idx < 0: continue

                dx2 = 1.0/self.dx**2
                dy2 = 1.0/self.dy**2
                center_val = 0.0

                cols = []
                vals = []

                if i > 0:
                    c = self.get_global_index(i-1, j, self.slot_elem)
                    cols.append(c); vals.append(-dx2); center_val += dx2
                if i < self.nx - 1:
                    c = self.get_global_index(i+1, j, self.slot_elem)
                    cols.append(c); vals.append(-dx2); center_val += dx2
                if j > 0:
                    c = self.get_global_index(i, j-1, self.slot_elem)
                    cols.append(c); vals.append(-dy2); center_val += dy2
                if j < self.ny - 1:
                    c = self.get_global_index(i, j+1, self.slot_elem)
                    cols.append(c); vals.append(-dy2); center_val += dy2

                cols.append(row_idx)
                vals.append(center_val)
                A.setValues([row_idx], cols, vals, PETSc.InsertMode.ADD_VALUES)

        # 2. Flux Equations (Identity)
        for j in range(ys, ye):
            for i in range(xs, xe + ex):
                idx_left = self.get_global_index(i, j, self.slot_left)
                if idx_left >= 0:
                    A.setValues([idx_left], [idx_left], [1.0], PETSc.InsertMode.ADD_VALUES)

        for j in range(ys, ye + ey):
            for i in range(xs, xe):
                idx_down = self.get_global_index(i, j, self.slot_down)
                if idx_down >= 0:
                    A.setValues([idx_down], [idx_down], [1.0], PETSc.InsertMode.ADD_VALUES)

        A.assemble()

        null_vec = self.dm.createGlobalVector()
        self.set_values(null_vec, func_elem=lambda x,y: 1.0, func_flux_x=lambda x,y: 0.0, func_flux_y=lambda x,y: 0.0)
        nullspace = PETSc.NullSpace().create(constant=False, vectors=[null_vec])
        A.setNullSpace(nullspace)
        return A

    def solve(self, A, rhs_vec):
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        # Use simple PCNONE to avoid singular matrix issues with direct solvers if singular
        ksp.setType('cg')
        ksp.getPC().setType('none')
        # Allow singular system (Neumann)
        ksp.setFromOptions()

        x = rhs_vec.duplicate()
        x.zeroEntries()
        ksp.solve(rhs_vec, x)
        return x

def main():
    PETSc.Sys.popErrorHandler()
    for nref in range(0,3):
        nx, ny = 10*2**nref, 10*2**nref
        solver = PoissonNeumannStag(nx, ny)

        PETSc.Sys.Print("Building DMStag Operators...")
        G, D = solver.build_operators()

        PETSc.Sys.Print("Building Laplacian A...")
        A = solver.assemble_laplacian()

        u_exact = solver.dm.createGlobalVector()
        rhs = solver.dm.createGlobalVector()

        def func_u(x, y):
            return np.cos(np.pi * x) * np.cos(np.pi * y)

        def func_f(x, y):
            return 2 * np.pi**2 * func_u(x, y)

        solver.set_values(u_exact, func_elem=func_u, func_flux_x=lambda x,y:0, func_flux_y=lambda x,y:0)
        solver.set_values(rhs, func_elem=func_f, func_flux_x=lambda x,y:0, func_flux_y=lambda x,y:0)

        PETSc.Sys.Print("Consistency check...")
        flux = G.createVecLeft()
        G.mult(u_exact, flux)
        div = D.createVecLeft()
        D.mult(flux, div)
        Au = A.createVecLeft()
        A.mult(u_exact, Au)
        Au.axpy(1.0, div)
        err_cons = Au.norm(PETSc.NormType.NORM_2)
        PETSc.Sys.Print(f"Consistency Error ||(A + D*G) u||: {err_cons:.4e}")

        PETSc.Sys.Print("Solving system...")
        u_sol = solver.solve(A, rhs)

        # Check Error using P matrix (AIJ) to mask flux
        P = solver.create_aij_matrix()
        (start_indices, widths, extra) = solver.dm.getCorners()
        xs, ys = start_indices[0], start_indices[1]
        xm, ym = widths[0], widths[1]
        xe, ye = xs + xm, ys + ym

        for j in range(ys, ye):
            for i in range(xs, xe):
                row_idx = solver.get_global_index(i, j, solver.slot_elem)
                if row_idx >= 0:
                    P.setValues([row_idx], [row_idx], [1.0], PETSc.InsertMode.ADD_VALUES)
                    P.assemble()

        u_sol_elem = u_sol.duplicate()
        P.mult(u_sol, u_sol_elem)

        u_exact_elem = u_exact.duplicate()
        P.mult(u_exact, u_exact_elem)

        norm_sol = u_sol_elem.norm()
        norm_ex = u_exact_elem.norm()

        sum_sol = u_sol_elem.sum()
        sum_ex = u_exact_elem.sum()
        mean_sol = sum_sol / (nx*ny)
        mean_ex = sum_ex / (nx*ny)
        u_sol_elem.shift(mean_ex - mean_sol)
        
        diff = u_sol_elem.duplicate()
        u_sol_elem.copy(diff)
        diff.axpy(-1.0, u_exact_elem)
        
        err_L2 = diff.norm(PETSc.NormType.NORM_2)
        PETSc.Sys.Print(f"L2 Error: {err_L2:.4e}")
        
        if err_L2 > 0.05:
            raise ValueError(f"L2 Error {err_L2:.4e} exceeds tolerance 0.05")

        P.destroy()

if __name__ == "__main__":
    main()
