import torch

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as f
from tqdm import tqdm
from itertools import islice
from pathlib import Path
import numpy as np

from meshtron import Meshtron
from tokenizer import Tokenizer2D
from dataset import MeshData


class Trainer:
    def __init__(
        self,
        data_path='../data/quad_data.pt',
        checkpoint_dir='checkpoints',
        verbose=True,
        quantization=1024,
        d_model=512,
        n_latents=None,
        batch_size=2,
        num_epochs=50,
        learning_rate=1e-4,
        stage_layers=[4, 8, 12, 16, 20],
        gradient_accumulation=None,
        max_val_samples=10,
        n_heads=8,
        window_size=None

    ):
        self.verbose = verbose
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.quantization = quantization
        self.d_model = d_model
        self.train_val_ratio = 0.80

        self.n_heads = n_heads
        self.gradient_accumulation = gradient_accumulation
        self.accumulator = 0
        self.window_size = window_size
        if n_latents == None:
            self.n_latents = self.d_model
        else:
            self.n_latents = n_latents

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.stage_layers = stage_layers
        self.data_path = data_path
        self.coord_dim = 2  # dimension of point_cloud coordinates
        self.max_val_samples = max_val_samples

        self.tokenizer = Tokenizer2D(
            quantization_levels=quantization, verbose=self.verbose)

        self.train_loader, self.val_loader, self.max_length, self.max_face_count, self.min_face_count = self.getData(
            self.train_val_ratio, self.data_path)

        if self.verbose == True:
            print('init meshtron')
        self.model = Meshtron(vocab_size=self.quantization+3,
                              d_model=self.d_model,
                              max_seq_length=self.max_length,
                              n_latents=self.n_latents,
                              input_dim=self.coord_dim,
                              min_face_count=self.min_face_count,
                              max_face_count=self.max_face_count,
                              n_heads=self.n_heads,
                              stage_layers=self.stage_layers,
                              verbose=self.verbose).to(self.device)

        if self.verbose == True:
            print('init optimizer')
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate)

        if self.verbose == True:
            print('init scheduler')
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, num_epochs)

        self.notation = f'q_{self.quantization}_d_model_{self.d_model}_n_latents_{self.n_latents}_batch_size_{self.batch_size}_n_heads_{self.n_heads}_window_size_{window_size}_stage_layers_' + \
            '_'.join(str(s) for s in self.stage_layers)

        self.checkpoint_dir = Path(checkpoint_dir) / self.notation
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 75
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }

        self.current_epoch = 0

        if True:

            print("\ntraining configuration")
            print(f"\ndevice: {self.device}")
            print(f"\nquantization: {self.quantization}")
            print(f"\nd_model: {self.d_model}")
            print(f"\nn_latents: {self.n_latents}")
            print(f"\nn_heads: {self.n_heads}")
            print(f"\nbatch_size: {self.batch_size}")
            print(f"\nnum_epochs: {self.num_epochs}")
            print(f"\nlearning_rate: {self.learning_rate}")
            print(f"\nstage_layers: {self.stage_layers}")

    def training(self):

        train_loss_history = []
        val_loss_history = []

        for i in tqdm(range(self.num_epochs), desc="epochs"):

            self.current_epoch = i
            current_train_loss = self.train_epoch()

            print('\nvalidation')
            current_val_loss = self.val_epoch()

            if self.best_val_loss > current_val_loss:
                self.best_val_loss = current_val_loss
                self.save_checkpoint()
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.training_history['train_loss'].append(
                current_train_loss)
            self.training_history['val_loss'].append(current_val_loss)
            self.training_history['epoch'].append(i)

            if self.patience_counter > self.max_patience:
                if self.verbose == True:
                    print(f'patience_counter reached after epochs {i}')
                break

            if self.verbose == True:
                print(
                    f'epoch: {i}, train loss: {current_train_loss}, validation loss: {current_val_loss}')
        if self.verbose == True:
            print(f'end of training')
        if i == self.num_epochs - 1:
            self.save_checkpoint()

    def train_epoch(self):

        self.model.train()

        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc='training')
        for batch in progress_bar:

            if self.window_size is None:

                if self.verbose == True:
                    if num_batches == 0 and self.current_epoch == 0:
                        print(f'window size is {self.window_size}')

                input_tokens = batch['input_tokens'].to(self.device)
                target_tokens = batch['target_tokens'].to(self.device)
                pad_token = batch['pad_token'][0].item()
                position_ids = None
            else:

                if self.verbose == True:
                    if num_batches == 0 and self.current_epoch == 0:
                        print(f'window size is {self.window_size}')

                num_tokens = batch['input_tokens'].size(1)

                max_int = int(np.floor((num_tokens-self.window_size)/8))
                rand_int = torch.randint(0, max_int, (1,))
                start_idx = 8*rand_int
                end_idx = 8*rand_int+self.window_size
                position_ids = torch.arange(start_idx.item(), end_idx.item(), device=self.device).unsqueeze(
                    0).expand(self.batch_size, -1)

                input_tokens = (batch['input_tokens']
                                [:, start_idx:end_idx]).to(self.device)
                target_tokens = (batch['target_tokens']
                                 [:, start_idx:end_idx]).to(self.device)
                pad_token = (batch['pad_token'][0]).item()

            face_count = batch['face_count'].to(self.device)
            point_cloud = batch['point_cloud'].to(self.device)
            # forward pass
            logits = self.model(input_tokens, point_cloud,
                                face_count, position_ids)
            reshaped_logits = logits.reshape(-1, logits.size(-1))
            reshaped_tokens = target_tokens.reshape(-1)

            # loss berechnen (ignorieren von pad (auffuell) tokens)

            loss = f.cross_entropy(
                reshaped_logits,
                reshaped_tokens,
                ignore_index=pad_token
            )

            if self.gradient_accumulation is None:
                if self.verbose == True:
                    if num_batches == 0 and self.current_epoch == 0:
                        print('\n no grad accumulation')
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            else:
                #
                if self.verbose == True:
                    if num_batches == 0 and self.current_epoch == 0:
                        print('\ngrad accumulation on')

                loss = loss / self.gradient_accumulation
                loss.backward()  # accumulate gradients

                self.accumulator += 1

                if self.accumulator >= self.gradient_accumulation:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.accumulator = 0

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / num_batches

    def val_epoch(self):

        self.model.eval()

        total_val_loss = 0

        num_batches = 0

        if self.max_val_samples is not None:
            progress_bar = tqdm(
                islice(self.val_loader, self.max_val_samples), total=self.max_val_samples, desc='validation')
        else:
            progress_bar = tqdm(self.val_loader, desc='validation')

        for batch in progress_bar:
            # for batch in self.train_loader:
            with torch.no_grad():

                if self.window_size is None:

                    if self.verbose == True:
                        if num_batches == 0 and self.current_epoch == 0:
                            print(f'window size is {self.window_size}')

                    input_tokens = batch['input_tokens'].to(self.device)
                    target_tokens = batch['target_tokens'].to(self.device)

                    position_ids = None
                else:

                    if self.verbose == True:
                        if num_batches == 0 and self.current_epoch == 0:
                            print(f'window size is {self.window_size}')

                    num_tokens = batch['input_tokens'].size(1)
                    max_int = int(np.floor((num_tokens-self.window_size)/8))
                    rand_int = torch.randint(0, max_int, (1,))
                    start_idx = 8*rand_int
                    end_idx = 8*rand_int+self.window_size
                    position_ids = torch.arange(start_idx.item(), end_idx.item(), device=self.device).unsqueeze(
                        0).expand(self.batch_size, -1)
                    input_tokens = (batch['input_tokens']
                                    [:, start_idx:end_idx]).to(self.device)
                    target_tokens = (batch['target_tokens']
                                     [:, start_idx:end_idx]).to(self.device)

                pad_token = (batch['pad_token'][0]).item()
                point_cloud = batch['point_cloud'].to(self.device)
                face_count = batch['face_count'].to(self.device)

               # forward pass
                logits = self.model(input_tokens, point_cloud,
                                    face_count, position_ids)
                reshaped_logits = logits.reshape(-1, logits.size(-1))
                reshaped_tokens = target_tokens.reshape(-1)

                # loss berechnen (ignorieren von pad (auffuell) tokens)

                val_loss = f.cross_entropy(
                    reshaped_logits,
                    reshaped_tokens,
                    ignore_index=pad_token
                )

                total_val_loss += val_loss.item()
                num_batches += 1

                # update progress bar
                progress_bar.set_postfix({'validation loss': val_loss.item()})

                if self.max_val_samples < num_batches:

                    print(f'end at num val samples {num_batches}')
                    return total_val_loss / num_batches

                input_tokens.detach()
                target_tokens.detach()
                pad_token
                point_cloud.detach()

        return total_val_loss / num_batches

    def getData(self, train_val_ratio, path='../data/quad_data.pt'):

        input_dim = 2
        train_size = train_val_ratio
        val_size = 1 - train_size

        meshes = torch.load(path, weights_only=False)

        max_faces = 0
        min_faces = float('inf')

        for mesh in meshes:
            num_faces = mesh.faces.size(1)
            if max_faces < num_faces:
                max_faces = num_faces
            if min_faces > num_faces:
                min_faces = num_faces

        max_token_sequence = max_faces*self.tokenizer.tokens_per_face + \
            2*self.tokenizer.n_start_end_tokens_repeat
        self.tokenizer.max_length_padding = max_token_sequence
        train_meshes, val_meshes = random_split(
            meshes,
            [train_size, val_size]
        )

        train_dataset = MeshData(
            train_meshes, self.tokenizer, verbose=self.verbose)
        val_dataset = MeshData(
            val_meshes, self.tokenizer, verbose=self.verbose)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        # collate_fn = collate_fn
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        max_train_length = train_dataset.max_seq_length
        max_val_length = val_dataset.max_seq_length

        max_length = 0
        if max_train_length < max_val_length:
            max_length = max_val_length
        else:
            max_length = max_train_length

        if self.verbose == True:
            print(f"max token sequenz length: {max_length}")

        return train_dataloader, val_dataloader, max_length, max_faces, min_faces

    def save_checkpoint(self):

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
        }

        best_path = f'{str(self.checkpoint_dir)}/epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, best_path)
        print(f"💾 Saved best model with val_loss: {self.best_val_loss:.4f}")

    def delete_gpu_memory(self):

        if self.verbose:
            print('\nPre Memory Delete')
            print(
                f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(
                f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        torch.cuda.empty_cache()

        if self.verbose:
            print('\nPost Memory Delete')
            print(
                f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(
                f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
