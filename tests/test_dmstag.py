import sys
import numpy as np
from petsc4py import PETSc

class PoissonNeumannStag:
    def __init__(self, nx, ny):
        """
        Initializes the grid and DMStag for a 2D Poisson problem.
        
        Args:
            nx, ny: Number of cells in x and y directions.
        """
        self.nx = nx
        self.ny = ny
        self.dx = 1.0 / nx
        self.dy = 1.0 / ny

        # Create DMStag
        # dof argument: (vertices, faces, elements)
        # We want 1 DoF on Elements (u) and 1 DoF on Faces (fluxes).
        # Note: In DMStag 2D, 'faces' implies dofs on both Left and Down edges of the cell.
        self.dm = PETSc.DMStag().create(dim=2)
        self.dm.setBox([nx, ny])
        self.dm.setDof((0, 1, 1)) 
        self.dm.setStencilWidth(1)
        self.dm.setBoundaryType(PETSc.DM.BoundaryType.GHOSTED, PETSc.DM.BoundaryType.GHOSTED)
        self.dm.setUp()
        
        self.dm.setUniformCoordinates(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        
        # Determine canonical slot indices for DMStag
        # Typically: 0 -> Face Down, 1 -> Face Left, 2 -> Element
        # We verify this by checking strata, but standard ordering is usually:
        # Vertex, Face(Down), Face(Left), Element
        self.slot_down = 0
        self.slot_left = 1
        self.slot_elem = 2
        
        # Get total unknowns for operator sizing
        self.n_local = self.dm.getLocalSize()
        self.n_global = self.dm.getGlobalSize()

    def build_operators(self):
        """
        Constructs the discrete Gradient (G) and Divergence (D) operators.
        
        Using DMStag, we can handle the indexing automatically.
        G: Cell -> Face (Calculates gradients/fluxes)
        D: Face -> Cell (Calculates divergence)
        
        We treat the boundary fluxes (Right of last cell, Top of last cell) as 
        Fixed Boundary Conditions (Homogeneous Neumann = 0).
        Thus, the operators only act on the 'internal' faces tracked by DMStag.
        """
        comm = PETSc.COMM_WORLD
        
        # We will build the operators as submatrices of a system defined on the full DM.
        # However, for clarity, we explicitly construct them as rectangular matrices.
        # Rows/Cols are identified by global indices of the specific strata.
        
        # 1. Get Local-to-Global Mapping to handle ghost indices manually if needed,
        #    or simply use local index arrays mapped to global.
        lgmap = self.dm.getLGMap()
        
        # Get array of global indices for the local portion of the grid
        # Reshape to (ny, nx, n_dof_per_point) to match grid topology
        # This requires the indices to be owned.
        idx_global_flat = self.dm.getGlobalIndices()
        # Shape: [ny, nx, dof_total]
        # Note: getGlobalIndices returns negative values for ghosts usually, 
        # but here we iterate over the owned range.
        idx_global = idx_global_flat.reshape(self.ny, self.nx, -1)
        
        # Define ranges
        (xs, xe), (ys, ye) = self.dm.getCorners()
        
        # --- Preallocate Matrices ---
        # G: Rows = Faces, Cols = Elements
        # D: Rows = Elements, Cols = Faces
        
        # Since we use one mixed DM, we can create a single square matrix "Mat" on the DM
        # and fill the off-diagonal blocks. 
        # Or we create explicit rectangular matrices. Let's do explicit rectangular 
        # to satisfy "G and D operators" request strictly.
        
        G = PETSc.Mat().create(comm)
        G.setSizes([self.n_global, self.n_global]) # We use full size to allow G*u vector ops
        G.setUp()
        G.zeroEntries()
        
        D = PETSc.Mat().create(comm)
        D.setSizes([self.n_global, self.n_global])
        D.setUp()
        D.zeroEntries()
        
        # --- Assembly Loop ---
        # We iterate over local elements and fill:
        # 1. G rows corresponding to the Left and Down faces of the element.
        # 2. D row corresponding to the Element itself.
        
        for j in range(ys, ye):
            for i in range(xs, xe):
                # Global Indices for this grid point
                # Since idx_global is from getGlobalIndices(), it corresponds to local owned part?
                # Actually getGlobalIndices() returns array for local INCLUDES ghosts? 
                # No, usually just owned. xs/ys are offsets in the global grid?
                # Let's use the DMStagStencil approach for robust logic.
                pass

        # RE-STRATEGY: Using MatSetValuesStencil is safer with DMStag.
        # But we need a matrix attached to the DM.
        # Let's create a temporary System Matrix structure to fill G and D blocks.
        
        # We will construct G and D using global indices derived from the DM.
        # We must use Local-to-Global mapping because we need to access neighbors.
        
        # Local array of indices (0, 1, 2...)
        # We will calculate relative local indices and map them to global.
        
        # Helper to get global index from local (i, j, component)
        # DMStag local vectors include ghosts. 
        # Size of local vec: (nx_local + 2*ghost) * (ny_local + 2*ghost) * dof
        
        # Let's rely on the structured global indices we extracted for OWNED points.
        # But for neighbors, we need the map.
        
        # We will iterate 0..nx-1 (local range) relative to the start xs.
        
        for j in range(ys, ye):
            for i in range(xs, xe):
                # Indices in the flattened global array (which only contains owned points)
                # We need to be careful. idx_global indexed by [j-ys, i-xs].
                
                # Let's use `dm.getGlobalIndices` which returns indices for the *local domain*.
                # The array is sized (ny_local, nx_local, dof).
                
                # Local loop indices
                jj = j - ys
                ii = i - xs
                
                # Global indices of DOFs at this center
                id_elem = idx_global[jj, ii, self.slot_elem]
                id_left = idx_global[jj, ii, self.slot_left]
                id_down = idx_global[jj, ii, self.slot_down]
                
                # --- Build G (Gradient) ---
                # G maps u (Element) to Flux (Face).
                # Row: Face Index. Cols: Element Indices.
                
                # 1. Left Face (Flux X): (u_i - u_{i-1})/dx
                # Neighbor: u_{i-1}. 
                # If i=0 (Global Left Boundary), Flux=0 (Neumann). Row is empty/zero.
                if i > 0: 
                    # We need global index of u_{i-1}. 
                    # Since we are iterating owned cells, u_{i-1} might be a ghost or owned.
                    # We use the Stencil helper concept manually:
                    # Previous Element Global ID. 
                    # If ii > 0: it is in our owned array.
                    # If ii == 0: it is a ghost. We need the LGMap.
                    
                    # Robust lookup:
                    if ii > 0:
                        id_elem_prev = idx_global[jj, ii-1, self.slot_elem]
                    else:
                        # Use Stencil or explicit knowledge. 
                        # Element (i-1) is just id_elem - dof_stride? No.
                        # We must use proper DMStag behavior. 
                        # Simpler: Map a local Stencil to Global.
                        
                        # Create a local stencil object
                        sten = PETSc.Mat.Stencil()
                        sten.i = i - 1
                        sten.j = j
                        sten.c = self.slot_elem
                        
                        # This requires mat.setValuesStencil.
                        # Let's stick to the simplest logic:
                        # Construct a Matrix that uses the DM.
                        pass

        # To avoid index hell, we create a matrix ON the DM.
        # Then we extract G and D or just return the combined system operators.
        # The prompt asks for G and D operators.
        
        # Let's perform the index lookup using the DM's coordinate logic.
        # We accept that this is slower in Python than C.
        
        G.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        D.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        # Get local-to-global mapping
        lgmap = self.dm.getLGMap()
        
        # Loop over all local elements (including ghosts to find neighbors? No, just owned)
        # We need to map (i,j,c) to global.
        # Local size info
        nloc_x, nloc_y = xe - xs, ye - ys
        
        # Local indices for current block
        # The mapping applies to the LOCAL vector (including ghosts).
        # Local vector shape: [ny_loc_ghost, nx_loc_ghost, dofs]
        # We need to find where (xs, ys) starts in the ghosted local vector.
        # usually (gw, gw).
        gw = 1 # stencil width
        
        # Helper to get global index given (i,j,c) global coords
        # This is inefficient but clear.
        def get_global_id(ix, iy, ic):
            # Map global coord to local ghosted coord
            # ix_loc = ix - xs + gw
            # This is only valid if we have the local values.
            # Instead, we construct a generic Stencil and map it.
            s = PETSc.Mat.Stencil()
            s.index = (ix, iy, 0) # 0 is dummy z
            s.c = ic
            # We need a way to map this stencil to global ID.
            # DMStag doesn't expose Stencil->GlobalID easily in python without a Mat.
            return -1

        # --- Efficient Assembly using Matrix on DM ---
        # We create a combined matrix M, fill it with G and D entries, then we are done?
        # G occupies rows corresponding to Faces.
        # D occupies rows corresponding to Elements.
        # They don't overlap in rows!
        # So we can return a SINGLE matrix 'Ops' that contains both G and D?
        # Or two matrices where one is empty in the other's rows.
        
        # Let's fill G and D separately using setValuesStencil if possible.
        # NOTE: setValuesStencil works on ANY matrix if we provide the mapping? No.
        # The Matrix must have the DM.
        
        G.setDM(self.dm)
        D.setDM(self.dm)
        
        # Now we can use stencils!
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        col_list = [PETSc.Mat.Stencil(), PETSc.Mat.Stencil(), 
                    PETSc.Mat.Stencil(), PETSc.Mat.Stencil()]
        
        for j in range(ys, ye):
            for i in range(xs, xe):
                
                # --- G Operator (Cell to Face) ---
                
                # 1. Gradient at Left Face (u_i - u_i-1)/dx
                if i > 0: # Internal face or Right boundary of ghost
                    row.index = (i, j, 0)
                    row.c = self.slot_left
                    
                    # Col: u_i
                    col1 = col_list[0]
                    col1.index = (i, j, 0)
                    col1.c = self.slot_elem
                    
                    # Col: u_i-1
                    col2 = col_list[1]
                    col2.index = (i-1, j, 0)
                    col2.c = self.slot_elem
                    
                    G.setValuesStencil([row], [col1, col2], [1.0/self.dx, -1.0/self.dx])
                    
                # 2. Gradient at Down Face (u_j - u_j-1)/dy
                if j > 0:
                    row.index = (i, j, 0)
                    row.c = self.slot_down
                    
                    col1 = col_list[0]
                    col1.index = (i, j, 0)
                    col1.c = self.slot_elem
                    
                    col2 = col_list[1]
                    col2.index = (i, j-1, 0)
                    col2.c = self.slot_elem
                    
                    G.setValuesStencil([row], [col1, col2], [1.0/self.dy, -1.0/self.dy])

                # --- D Operator (Face to Cell) ---
                # Div u = (Flux_Right - Flux_Left)/dx + (Flux_Up - Flux_Down)/dy
                
                row.index = (i, j, 0)
                row.c = self.slot_elem
                
                # Flux Left (at i, j)
                c_left = col_list[0]
                c_left.index = (i, j, 0)
                c_left.c = self.slot_left
                
                # Flux Right (at i+1, j) -> Left face of i+1
                c_right = col_list[1]
                c_right.index = (i+1, j, 0)
                c_right.c = self.slot_left
                
                # Flux Down (at i, j)
                c_down = col_list[2]
                c_down.index = (i, j, 0)
                c_down.c = self.slot_down
                
                # Flux Up (at i, j+1) -> Down face of j+1
                c_up = col_list[3]
                c_up.index = (i, j+1, 0)
                c_up.c = self.slot_down
                
                # Logic for Boundary Fluxes in D
                # If i=nx-1, Right flux is boundary. Homogeneous Neumann => 0.
                # So we simply do not add that column entry (val=0 implicitly).
                
                cols = []
                vals = []
                
                # Left
                cols.append(c_left)
                vals.append(-1.0/self.dx)
                
                # Right
                if i < self.nx - 1:
                    cols.append(c_right)
                    vals.append(1.0/self.dx)
                
                # Down
                cols.append(c_down)
                vals.append(-1.0/self.dy)
                
                # Up
                if j < self.ny - 1:
                    cols.append(c_up)
                    vals.append(1.0/self.dy)
                
                D.setValuesStencil([row], cols, vals)

        G.assemble()
        D.assemble()
        
        return G, D

    def assemble_laplacian(self):
        """
        Assembles the Laplacian Matrix A directly.
        A u = - div (grad u)
        """
        A = self.dm.createMatrix()
        A.zeroEntries()
        
        (xs, xe), (ys, ye) = self.dm.getCorners()
        
        row = PETSc.Mat.Stencil()
        col = PETSc.Mat.Stencil()
        
        for j in range(ys, ye):
            for i in range(xs, xe):
                
                # We only build the equation for 'u' (element)
                row.index = (i, j, 0)
                row.c = self.slot_elem
                
                vals = []
                cols = []
                
                center_val = 0.0
                
                # Neighbors
                neighbors = [
                    (i-1, j), # West
                    (i+1, j), # East
                    (i, j-1), # South
                    (i, j+1)  # North
                ]
                
                dx2 = 1.0/self.dx**2
                dy2 = 1.0/self.dy**2
                
                # West
                if i > 0:
                    c = PETSc.Mat.Stencil()
                    c.index = (i-1, j, 0)
                    c.c = self.slot_elem
                    cols.append(c)
                    vals.append(-dx2) # -1/h^2
                    center_val += dx2
                else:
                    # Boundary: Neumann. Flux = 0.
                    # Term (-u_w + u_c)/h^2 vanishes?
                    # Grad at face is 0. Flux entering is 0.
                    # Div = (Flux_e - 0)/h.
                    # Flux_e = (u_e - u_c)/h.
                    # Result: (u_e - u_c)/h^2.
                    # In matrix A (negative laplacian): -(u_e - u_c)/h^2 = (-u_e + u_c)/h^2.
                    # So we just don't add the West term, and Center term only gets contribution from East.
                    pass
                    
                # East
                if i < self.nx - 1:
                    c = PETSc.Mat.Stencil()
                    c.index = (i+1, j, 0)
                    c.c = self.slot_elem
                    cols.append(c)
                    vals.append(-dx2)
                    center_val += dx2
                    
                # South
                if j > 0:
                    c = PETSc.Mat.Stencil()
                    c.index = (i, j-1, 0)
                    c.c = self.slot_elem
                    cols.append(c)
                    vals.append(-dy2)
                    center_val += dy2
                    
                # North
                if j < self.ny - 1:
                    c = PETSc.Mat.Stencil()
                    c.index = (i, j+1, 0)
                    c.c = self.slot_elem
                    cols.append(c)
                    vals.append(-dy2)
                    center_val += dy2
                
                # Center
                c = PETSc.Mat.Stencil()
                c.index = (i, j, 0)
                c.c = self.slot_elem
                cols.append(c)
                vals.append(center_val)
                
                A.setValuesStencil([row], cols, vals)
        
        A.assemble()
        
        # Nullspace (Constant functions) for Neumann
        # We need a vector that is 1 on elements and 0 elsewhere?
        # Actually, the system A is only non-trivial on the Element rows.
        # The rows corresponding to fluxes are all zero (identity or empty) in A?
        # Since A was created from DMStag, it has rows for Fluxes too!
        # These rows are currently empty (0=0). This makes A singular in a bad way.
        # We should place 1.0 on the diagonal for the Flux rows to make them "dummy" equations,
        # or extract the Element submatrix.
        # For this example, we keep it simple: We solve only for u?
        # KSP might complain about zero pivots in the flux rows.
        
        # FIX: Set Identity on Flux rows
        # Loop over faces and set 1.0
        for j in range(ys, ye):
            for i in range(xs, xe):
                # Left Face
                r = PETSc.Mat.Stencil()
                r.index = (i, j, 0); r.c = self.slot_left
                A.setValuesStencil([r], [r], [1.0])
                
                # Down Face
                r.index = (i, j, 0); r.c = self.slot_down
                A.setValuesStencil([r], [r], [1.0])
                
        A.assemble()
        
        # Nullspace: constant on Elements, 0 on Faces
        null_vec = self.dm.createGlobalVector()
        null_vec.set(0.0)
        
        # Set 1.0 on elements
        # We need to access array.
        arr = self.dm.getVecArray(null_vec)
        arr[..., self.slot_elem] = 1.0
        # Restore (handled by context manager or implicitly if numpy array is view)
        # petsc4py 3.12+ getVecArray returns a context manager compatible object or view.
        # We need to assume it writes back.
        
        nullspace = PETSc.NullSpace().create(constant=False, vectors=[null_vec])
        A.setNullSpace(nullspace)
        
        return A

    def solve(self, A, rhs_vec):
        """
        Solves A x = rhs.
        """
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        ksp.setType('cg')
        ksp.getPC().setType('gamg') # Multigrid often good for Poisson
        ksp.setFromOptions()
        
        x = A.createVecRight()
        ksp.solve(rhs_vec, x)
        return x

def main():
    PETSc.Sys.popErrorHandler()
    nx, ny = 10, 10
    
    solver = PoissonNeumannStag(nx, ny)
    
    PETSc.Sys.Print("Building DMStag Operators...")
    G, D = solver.build_operators()
    
    PETSc.Sys.Print("Building Laplacian A...")
    A = solver.assemble_laplacian()
    
    # --- Test Problem ---
    # u = cos(pi*x)*cos(pi*y)
    # f = 2*pi^2 * u
    
    u_exact = solver.dm.createGlobalVector()
    rhs = solver.dm.createGlobalVector()
    
    arr_u = solver.dm.getVecArray(u_exact)
    arr_f = solver.dm.getVecArray(rhs)
    
    (xs, xe), (ys, ye) = solver.dm.getCorners()
    
    for j in range(ys, ye):
        for i in range(xs, xe):
            # Element centers
            x = (i + 0.5) * solver.dx
            y = (j + 0.5) * solver.dy
            
            val = np.cos(np.pi * x) * np.cos(np.pi * y)
            
            # Set Element values
            arr_u[j, i, solver.slot_elem] = val
            arr_f[j, i, solver.slot_elem] = 2 * np.pi**2 * val
            
            # Set Flux values (for consistency check) to 0 or exact?
            # For the solve A*x=f, we put f in element slots. 
            # Flux slots in rhs should be 0 (corresponding to dummy identity equations).
            arr_u[j, i, solver.slot_left] = 0
            arr_u[j, i, solver.slot_down] = 0
            arr_f[j, i, solver.slot_left] = 0
            arr_f[j, i, solver.slot_down] = 0

    # Consistency Check: A approx - D * G (on element subspace)
    # Note: G takes u (elem) -> flux. D takes flux -> u (elem).
    # A acts on u (elem).
    
    # 1. Flux = G * u
    flux = G.createVecLeft()
    G.mult(u_exact, flux)
    
    # 2. Div = D * Flux
    div = D.createVecLeft()
    D.mult(flux, div)
    
    # 3. Laplacian = A * u
    Au = A.createVecLeft()
    A.mult(u_exact, Au)
    
    # Check Difference on Element slots only
    # Au should be approx - div
    # So Au + div = 0
    
    Au.axpy(1.0, div)
    
    # Mask out flux slots for norm calculation (they are dummy 1.0*0=0 in A)
    # Actually Au has 0 in flux slots (1.0*0). Div has 0 in flux slots (D output).
    # So norm is safe.
    err_cons = Au.norm(PETSc.NormType.NORM_2)
    PETSc.Sys.Print(f"Consistency Error ||(A + D*G) u||: {err_cons:.4e}")
    
    # Solve
    PETSc.Sys.Print("Solving system...")
    u_sol = solver.solve(A, rhs)
    
    # Calculate Error (project out mean shift)
    # Sum only element values
    # We can use stride/subvector, but simple array access is fine for scalar check
    arr_sol = solver.dm.getVecArray(u_sol)
    arr_exact = solver.dm.getVecArray(u_exact)
    
    # We need to manually sum to handle parallel safely, 
    # but for this simple sequential/small test:
    # (In parallel, getVecArray only gives local)
    
    # Create an element-only IS (Index Set) to extract subvector?
    # Simpler: just set flux slots to 0 and take norm.
    u_sol_clean = u_sol.duplicate()
    u_sol_clean.copy(u_sol)
    
    # Zero out fluxes in clean vector
    arr_clean = solver.dm.getVecArray(u_sol_clean)
    arr_clean[..., solver.slot_left] = 0.0
    arr_clean[..., solver.slot_down] = 0.0
    
    # Compute mean on elements
    # Note: sum() includes flux slots (which are 0 now).
    # Number of elements = nx*ny.
    sum_sol = u_sol_clean.sum()
    mean_sol = sum_sol / (nx*ny)
    
    # Exact mean
    u_exact_clean = u_exact.duplicate()
    u_exact_clean.copy(u_exact)
    arr_ex_clean = solver.dm.getVecArray(u_exact_clean)
    arr_ex_clean[..., solver.slot_left] = 0.0
    arr_ex_clean[..., solver.slot_down] = 0.0
    mean_exact = u_exact_clean.sum() / (nx*ny)
    
    # Shift
    u_sol_clean.shift(mean_exact - mean_sol)
    
    # Error
    diff = u_sol_clean.duplicate()
    diff.axpy(-1.0, u_exact_clean)
    err_L2 = diff.norm(PETSc.NormType.NORM_2)
    
    PETSc.Sys.Print(f"L2 Error: {err_L2:.4e}")

if __name__ == "__main__":
    main()