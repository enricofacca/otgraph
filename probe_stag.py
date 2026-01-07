from petsc4py import PETSc
import numpy as np

def probe_dmstag():
    nx, ny = 2, 2
    dm = PETSc.DMStag().create(dim=2)
    dm.setGlobalSizes([nx, ny])
    dm.setDof((0, 1, 1))
    dm.setStencilWidth(1)
    dm.setBoundaryTypes([PETSc.DM.BoundaryType.GHOSTED, PETSc.DM.BoundaryType.GHOSTED])
    dm.setUp()

    dm.setCoordinateDMType(PETSc.DM.Type.STAG)
    dm.setUniformCoordinates(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

    # Slots
    slot_down = dm.getLocationSlot(PETSc.DMStag.StencilLocation.DOWN, 0)
    slot_left = dm.getLocationSlot(PETSc.DMStag.StencilLocation.LEFT, 0)
    slot_elem = dm.getLocationSlot(PETSc.DMStag.StencilLocation.ELEMENT, 0)

    print(f"Slots: Down={slot_down}, Left={slot_left}, Elem={slot_elem}")

    # Layout
    # entries_per_element = dm.getEntriesPerElement() # 3
    # Local sizes (excluding ghosts)
    (start_indices, widths, extra) = dm.getCorners()
    xs, ys = start_indices[0], start_indices[1]
    xm, ym = widths[0], widths[1]
    xe, ye = xs + xm, ys + ym
    # widths = (xe-xs, ye-ys)
    nx_local = xe - xs
    ny_local = ye - ys

    # LGMap
    lgmap = dm.getLGMap()
    indices = lgmap.getIndices() # Local to Global mapping array
    # indices[k] gives global index of local index k

    # Local representation includes ghosts?
    # LGMap usually maps "Local Vector" indices (including ghosts) to Global Indices.
    # But LGMap size might be smaller if it only covers owned?
    # check size
    print(f"LGMap size: {indices.size}")

    # Local Vector size
    lvec = dm.createLocalVector()
    print(f"Local Vec size: {lvec.getSize()}")

    # We expect indices.size == lvec.getSize() roughly, or at least coverage for owned.

    # Let's verify the mapping formula:
    # Local Index for (i, j, slot)
    # i, j are global coordinates.
    # Map to local coordinates relative to ghosted region.
    # (start_indices, widths, extra) = dm.getCorners()
    # But we need ghost corners to know local offsets.
    (gxs, gxe), (gys, gye) = dm.getGhostCorners()
    print(f"Ghost Corners: x=({gxs}, {gxe}), y=({gys}, {gye})")

    # Local coordinates:
    # i_loc = i - gxs
    # j_loc = j - gys
    # width_loc = gxe - gxs
    # idx_loc = (j_loc * width_loc + i_loc) * entries_per_elem + slot

    # Let's test this hypothesis by creating a global vector, setting a value at a known global index,
    # and checking where it appears?
    # No, we want to SET values using global indices.
    # So we need to calculate Global Index for (i, j, slot).

    # Global Index = lgmap.getIndices()[idx_loc] ?
    # Let's check.

    entries_per_elem = 3
    width_loc_x = gxe - gxs

    print("Testing Mapping:")
    for j in range(ys, ye):
        for i in range(xs, xe):
            # Check slot_elem
            i_loc = i - gxs
            j_loc = j - gys
            idx_loc = (j_loc * width_loc_x + i_loc) * entries_per_elem + slot_elem

            global_idx = indices[idx_loc]
            print(f"Elem ({i},{j}) -> Local {idx_loc} -> Global {global_idx}")

probe_dmstag()
