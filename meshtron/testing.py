
import numpy as np
import matplotlib.pyplot as plt
import plotting_tools
from trainer import Trainer
import torch
from tokenizer import Tokenizer2D
from tqdm import tqdm
from collections import defaultdict


# Initialisiere die Variable mit einem sehr großen Wert
# path0 = '../data/quad_data.pt'

path_meshes = '../data/structured_quad_meshes_pre_selected.pt'

quantization = 1024
d_model = 1024
n_latents = 512
batch_size = 16
num_epochs = 500
learning_rate = 1e-4
stage_layers = [4, 4, 4, 4, 4]
window_size = 600
gradient_accumulation = 2
n_heads = 8
epoch_saving_point = 20

verbose = False
trainer = Trainer(data_path=path_meshes, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size, quantization=quantization, d_model=d_model, n_latents=n_latents,
                  gradient_accumulation=gradient_accumulation, window_size=window_size, n_heads=n_heads, stage_layers=stage_layers, verbose=verbose)

# path = f'./checkpoints/saved_trainings/q_{quantization}_d_model_{d_model}_n_latents_{n_latents}_batch_size{batch_size}_stage_layers_' + \
#     '_'.join(str(s) for s in stage_layers)+f'/epoch_{epoch_saving_point}.pt'
#
path = f'./checkpoints/q_{quantization}_d_model_{d_model}_n_latents_{n_latents}_batch_size_{batch_size}_n_heads_{n_heads}_window_size_{window_size}_stage_layers_' + \
    '_'.join(str(s) for s in stage_layers)+f'/epoch_{epoch_saving_point}.pt'


checkpoint = torch.load(path, weights_only=True)

state = checkpoint['model_state_dict']
trainer.model.load_state_dict(state)

data = next(iter(trainer.val_loader))

input_tokens = data['input_tokens'][0:1]
target_tokens = data['target_tokens'][0:1]
point_cloud = data['point_cloud'][0:1]
face_count = data['face_count'][0:1]
start = input_tokens[0:1, 0:8]
face_count
point_cloud.size()
n_faces = face_count
# n_faces[0] = 24

generated_tokens = trainer.model.generate(
    point_cloud=point_cloud,
    face_count=n_faces,
    start_tokens=start,
    max_length=1500
)


vertices, quads = trainer.tokenizer.detokenize(generated_tokens)
plotting_tools.plt_mesh(
    vertices, quads, point_cloud=point_cloud[0], output_file='./figures/test_mesh.png')


quads.size()

vertices_true, quads_true = trainer.tokenizer.detokenize(input_tokens[0])
plotting_tools.plt_mesh(vertices_true, quads_true,
                        output_file='./figures/true_mesh.png')

plotting_tools.plt_mesh(mesh.x[:, 0:2], mesh.faces,
                        output_file='./true_mesh.png')
plotting_tools.plt_mesh(vertices, quads, point_cloud=points,
                        output_file='./test_mesh.png')


def main():

    path_meshes = '../data/structured_quad_meshes_pre_selected.pt'

    quantization = 2048
    d_model = 512
    n_latents = 2*d_model
    batch_size = 6
    num_epochs = 75
    learning_rate = 1e-4
    stage_layers = [2, 2, 2, 2, 2]
    window_size = None
    gradient_accumulation = 2
    n_heads = 8

    self.trainer = Trainer(data_path=path_meshes, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size, quantization=quantization, d_model=d_model, n_latents=n_latents,
                           gradient_accumulation=gradient_accumulation, window_size=window_size, n_heads=n_heads, stage_layers=stage_layers)

    self.trainer.training()


if __name__ == '__main__':
    main()
