import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from tqdm import tqdm

from meshtron import Meshtron
from tokenizer import Tokenizer2D
from dataset import MeshData


path = '../data/quad_data.pt'
meshes = torch.load(path)


def train_epoch(model, train_dataloader,  optimizer, device='cuda'):

    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc='training')
    for batch in progress_bar:
        # daten auf device
        input_tokens = batch['input_tokens'].to(device)
        target_tokens = batch['target_tokens'].to(device)
        pad_token = batch['pad_token'].to(device)
        point_cloud = batch['point_cloud'].to(device)

        # forward pass
        logits = model(input_tokens, point_cloud)
        reshaped_logits = logits.reshape(-1, logits.size(-1)),
        reshaped_tokens = target_tokens.reshape(-1)
        # loss berechnen (ignorieren von pad (auffuell) tokens)

        loss = f.cross_entropy(
            reshaped_logits,
            reshaped_tokens,
            ignore_index=pad_token
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)  # gradient clipping
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # update progress bar
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches


def main():


    verbose = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    quantization = 1024
    d_model = 512
    n_latents = d_model
    batch_size = 2
    num_epochs = 50
    learning_rate = 1e-4
    stage_layers = [4, 8, 12, 16, 20]

    input_dim = 2  # dimension of point_cloud coordinates
    train_size = 0.8
    val_size = 1 - train_size

    if verbose == True:

        print("\ntraining configuration")
        print(f"\ndevice: {device}")
        print(f"\nquantization: {quantization}")
        print(f"\nd_model: {d_model}")
        print(f"\nn_latents: {n_latents}")
        print(f"\nbatch_size: {batch_size}")
        print(f"\nnum_epochs: {num_epochs}")
        print(f"\nlearning_rate: {learning_rate}")
        print(f"\nstage_layers: {stage_layers}")

    # initialisiere tokenizer
    tokenizer = Tokenizer2D(quantization_levels=quantization)
    train_meshes, val_meshes = random_split(
        meshes,
        [train_size, val_size]
    )

    train_dataset = MeshData(train_meshes, tokenizer)
    val_dataset = MeshData(val_meshes, tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    max_train_length = train_dataset.max_seq_length
    max_val_length = val_dataset.max_seq_length

    max_length = 0
    if max_train_length < max_val_length:
        max_length = max_val_length
    else:
        max_length = max_train_length

    if verbose == True:
        print(f"max token sequenz length: {max_length}")

    meshtron = Meshtron(vocab_size=quantization+3,
                        d_model=d_model,
                        max_seq_length=max_length,
                        n_latents=n_latents,
                        input_dim=input_dim).to(device)

    for batch in train_dataloader:
        print(batch)
        break


    dataset = train_dataset
    i = 0

    in_tokens = dataset[i]['input_tokens']
    target_tokens = dataset[i]['target_tokens']

    point_cloud = dataset[i]['point_cloud']

    point_cloud = point_cloud.unsqueeze(0)
    in_tokens = in_tokens.unsqueeze(0)
    target_tokens = target_tokens.unsqueeze(0)

    output = meshtron.forward(in_tokens.to(device), point_cloud.to(device))
    reshape_out = output.reshape(-1, output.size(-1))
    reshape_out.size()

    out_loss = output.reshape(-1, output.size(-1))
    target_loss = target_tokens.reshape(-1)
    out_loss.size()
    target_loss.size()

    loss = F.cross_entropy(
        out_loss.to(device),
        target_loss.to(device),
        reduction='none'
    )

    loss.size()
    probabilities = torch.sum(out_loss, dim=1)
    probabilities.size()    # Konfiguration
    optimizer = optim.AdamW(meshtron.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # Training Loop
    if verbose == True:
        print("\nStarte Training...")

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        # Train
        avg_loss = train_epoch(meshtron, train_dataloader,
                               val_dataloader, optimizer, device)
        print(f"Average Loss: {avg_loss:.4f}")

        # Learning Rate Schedule
        scheduler.step()

        # Speichere Checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pt')
            print(f"Checkpoint gespeichert!")

            # Teste Generierung
            print("\nTeste Generierung...")
            model.eval()
            with torch.no_grad():
                # Nimm eine zufällige Point Cloud
                sample_idx = np.random.randint(len(dataset))
                sample = dataset[sample_idx]
                point_cloud = sample['point_cloud'].unsqueeze(0).to(device)

                # Generiere
                generated_tokens = model.generate(
                    point_cloud=point_cloud,
                    max_length=100,
                    temperature=0.8,
                    device=device
                )

                print(
                    f"Generierte Token-Sequenz (erste 20): {generated_tokens[:20].tolist()}")

    print("\nTraining abgeschlossen!")

    # Speichere finales Modell
    torch.save(model.state_dict(), 'final_model.pt')
