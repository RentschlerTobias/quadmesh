import torch


class MeshCheck:
    def __init__(self, tri_mesh, quad_mesh, tol=1e-6, rel_tol=1e-2):

        tri_vertices = tri_mesh.x[:, 0:2]
        tri_faces = tri_mesh.faces.T
        tri_nodes = tri_vertices[tri_faces]
        A = tri_nodes[:, 0, :]
        B = tri_nodes[:, 1, :]
        C = tri_nodes[:, 2, :]
        AB = B - A
        AC = C - A
        cross_product = AB[:, 0] * AC[:, 1] - AB[:, 1] * AC[:, 0]
        self.tri_area = torch.sum(torch.abs(cross_product) / 2.0)
       #
        quad_vertices = quad_mesh.x[:, 0:2]
        quad_faces = quad_mesh.faces.T
        quad_nodes = quad_vertices[quad_faces]
        A = quad_nodes[:, 0, :]
        B = quad_nodes[:, 1, :]
        C = quad_nodes[:, 2, :]
        D = quad_nodes[:, 3, :]

        diag1 = C - A
        diag2 = D - B
        cross_product = diag1[:, 0] * diag2[:, 1] - diag1[:, 1] * diag2[:, 0]
        self.quad_area = torch.sum(torch.abs(cross_product) / 2.0)
        # self.quad_area = torch.sum(triangle1_area + triangle2_area)
        # self.quad_area = torch.sum(torch.abs(cross1 + cross2) / 2.0)

        # Relative area match. Absolute tol on a variable-size domain rejected
        # topologically-correct partitions whose quad edges bow (spline curves)
        # by a tiny absolute amount. Compare relative to the tri reference area.
        abs_diff = torch.abs(self.tri_area - self.quad_area)
        rel_diff = abs_diff / (torch.abs(self.tri_area) + 1e-12)
        self.abs_area_diff = abs_diff
        self.rel_area_diff = rel_diff

        if abs_diff <= tol or rel_diff <= rel_tol:
            self.is_valid = True
        else:
            self.is_valid = False
