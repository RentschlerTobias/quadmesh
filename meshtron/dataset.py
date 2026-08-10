from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm


class MeshData(Dataset):

    """
    Dataset für Meshtron 
    """

    def __init__(self, meshes, tokenizer, max_seq_length=None, n_sample_points=1500, verbose=True, boundary_points_only=False):
        """
        meshes: Liste von Mesh-Objekten 
        tokenizer:  Tokenizer2D
        max_seq_length: Maximale Sequenzlänge (für Padding)
        """
        self.meshes = meshes
        self.tokenizer = tokenizer
        self.data = []
        self.point_clouds = []
        self.face_count = []
        self.boundary_points_only = boundary_points_only
        if verbose == True:
            print(f"start tokenizing")
        for i in tqdm(range(len(meshes)), desc="meshes"):
            mesh = meshes[i]
            vertices = mesh.x[:, 0:2]  # 2D vertices
            faces = mesh.faces

            tokens = tokenizer.tokenize(vertices, faces)
            point_cloud = self.get_point_cloud(mesh, n_sample_points)
            num_faces = faces.size(1)

            self.data.append(tokens)

            self.point_clouds.append(point_cloud)
            self.face_count.append(num_faces)
        if max_seq_length is None:
            self.max_seq_length = max(len(tokens) for tokens in self.data)
        else:
            self.max_seq_length = max_seq_length

        self.min_seq_length = min(len(tokens) for tokens in self.data)
        print(
            f"\nMax Sequenzlänge: {self.max_seq_length}\nMin Sequenzlänge: {self.min_seq_length}")

    def get_point_cloud(self, mesh, n_sample_points):

        mask = mesh.tri_coordinates[:, 2] != 2

        boundary_points = mesh.tri_coordinates[mask, 0:2]
        interior_points = mesh.tri_coordinates[~mask, 0:2]

        num_boundary_points = boundary_points.size(0)
        num_interior_points = interior_points.size(0)

        remaining = n_sample_points - num_boundary_points

        point_cloud = torch.ones([n_sample_points, 2]) * (-1)
        if num_boundary_points >= n_sample_points:
            # Sample nur aus Boundary Points
            random_idx = torch.randint(
                0, num_boundary_points, [n_sample_points])
            point_cloud[:, :] = boundary_points[random_idx, :]

        else:

            points = []
            points.append(boundary_points)

            if num_interior_points >= remaining:

                random_idx = torch.randint(
                    0, num_interior_points, (remaining,)
                )
                points.append(interior_points[random_idx, :])

            else:
                repeat_factor = (remaining + num_interior_points -
                                 1) // num_interior_points  # ceil division
                interior_repeated = interior_points.repeat(repeat_factor, 1)

                noise_strength = 0.001
                noise = torch.rand_like(interior_repeated)*noise_strength
                noisy_interior_points = interior_repeated + noise
                num_noisy_points = noisy_interior_points.size(0)

                random_idx = torch.randint(
                    0, num_noisy_points, (remaining,)
                )
                points.append(noisy_interior_points[random_idx, :])

            point_cloud = torch.cat(points, dim=0)

        return point_cloud

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        point_cloud = self.point_clouds[idx]
        num_faces = self.face_count[idx]

        pad_token = self.tokenizer.pad_token

        # Erstelle Input und Target für autoregressives Training
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]

        # Padding (auffuellen der token sequencen, damit alle die gleiche laenge haben)
        tokens_padded = tokens + [pad_token] * \
            (self.max_seq_length - len(tokens))
        tokens_tensor = torch.LongTensor(tokens_padded)

        # Input: alle außer letztes Token, Target: alle außer erstes Token
        input_tokens = tokens_tensor[:-1]
        target_tokens = tokens_tensor[1:]

        # Erstelle Padding Mask (wo NICHT gepaddet ist = 1)
        padding_mask = (input_tokens != pad_token).float()

        return {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
            'padding_mask': padding_mask,
            'pad_token': pad_token,
            'point_cloud': torch.FloatTensor(point_cloud),
            'face_count': num_faces,
            'seq_length': len(tokens) - 1  # Tatsächliche Länge ohne Padding
        }
