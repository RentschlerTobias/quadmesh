
from trainer import Trainer
import torch


def main():

    # path_meshes = '../data/unstructured_quad_meshes_v2.pt'
    path_meshes = '../data/structured_quad_meshes_pre_selected.pt'

    quantization = 2048
    d_model = 512
    n_latents = 1024
    batch_size = 18
    num_epochs = 500
    learning_rate = 1e-4
    stage_layers = [4, 4, 4, 4, 4]
    window_size = 600
    gradient_accumulation = 2
    n_heads = 8
    verbose = False

    trainer = Trainer(data_path=path_meshes, num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size, quantization=quantization, d_model=d_model, n_latents=n_latents,
                      gradient_accumulation=gradient_accumulation, window_size=window_size, n_heads=n_heads, stage_layers=stage_layers, verbose=verbose)

    trainer.training()


if __name__ == "__main__":
    main()
