import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import math
import random
import pandas as pd
import seaborn as sns


def load_data():
    base_path = "./checkpoints/"

    trained_model_path = os.listdir(base_path)

    history = []

    for trained_model in trained_model_path:

        if trained_model == 'saved_trainings':
            continue

        path = base_path + trained_model

        checkpoints = os.listdir(path)
        directories = [path + '/' + checkpoint for checkpoint in checkpoints]

        # checkpoints_path = path + '/' + chekpoints
        last_checkpoint = max(directories, key=os.path.getmtime)

        model_params = extract_params(last_checkpoint)

        saved_train_data = torch.load(last_checkpoint)
        history.append([saved_train_data["training_history"], model_params])

    return history


def extract_params(dir_name):
    params = {}
    # Extrahiere alle einfachen Parameter (name_wert)
    params['q'] = int(re.search(r'q_(\d+)', dir_name).group(1))
    params['d_model'] = int(re.search(r'd_model_(\d+)', dir_name).group(1))
    params['n_latents'] = int(re.search(r'n_latents_(\d+)', dir_name).group(1))
    params['batch_size'] = int(
        re.search(r'batch_size_(\d+)', dir_name).group(1))
    params['n_heads'] = int(re.search(r'n_heads_(\d+)', dir_name).group(1))

    # Window size (kann None sein)
    window_match = re.search(r'window_size_(None|\d+)', dir_name)
    params['window_size'] = None if window_match.group(
        1) == 'None' else int(window_match.group(1))

    # Stage layers (Liste von Zahlen)
    stage_layers_match = re.search(r'stage_layers_([\d_]+)', dir_name)
    params['stage_layers'] = [int(x)
                              for x in stage_layers_match.group(1).split('_')]
    return params


def plot_losses(history, output_file='./figures/loss_history_3D.png'):
    figsize = (15, 15)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    for i, data in enumerate(history):
        color = (random.random(), random.random(), random.random())
        loss = data[0]
        params = data[1]
        train_loss = np.array(loss['train_loss'])
        val_loss = np.array(loss['val_loss'])
        epoch = np.array(loss['epoch'])

        z = np.ones(epoch.size) * 5*i  # Z-Wert für diesen Run

        ax.scatter(epoch, z, train_loss, s=5, marker='o', c=[color], zorder=10)
        ax.scatter(epoch, z, val_loss, s=5, marker='x', c=[color], zorder=10)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Run Number')
    ax.set_zlabel('Loss')
    ax.view_init(elev=20, azim=45)  # Viewing angle anpassen

    plt.savefig(output_file, dpi=600, transparent=True)


def plot_loss_by_params(history, output_file='./figures/loss_history_params.png'):

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    params_to_check = ['q', 'd_model', 'n_latents',
                       'n_heads', 'window_size', 'stage_layers']
    n_params = len(params_to_check)

    if n_params % 2 < n_params % 3:
        n_cols = 2
    else:
        n_cols = 3

    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

# Flatten axes array für einfacheren Zugriff
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    for idx, param_name in enumerate(params_to_check):
        ax = axes[idx]

        # Sammle alle einzigartigen Parameterwerte
        if param_name == 'stage_layers':
            # Konvertiere Listen zu Tupeln für Hashbarkeit
            unique_values = list(
                set(tuple(data[1][param_name]) for data in history))
            # Sortiere Tupel
            unique_values.sort()
        else:
            unique_values = list(set(data[1][param_name] for data in history))
            # Sortiere mit None-Handling
            unique_values.sort(key=lambda x: (x is None, x))

        # Erstelle Farbmap mit besserer Unterscheidbarkeit
        if len(unique_values) <= 10:
            colors = plt.cm.tab10(np.arange(len(unique_values)))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_values)))

        color_map = {val: colors[i] for i, val in enumerate(unique_values)}

        for data in history:
            loss = data[0]
            params = data[1]

            train_loss = np.array(loss['train_loss'])
            val_loss = np.array(loss['val_loss'])
            epoch = np.array(loss['epoch'])

            # Konvertiere zu Tupel falls stage_layers
            if param_name == 'stage_layers':
                param_value = tuple(params[param_name])
            else:
                param_value = params[param_name]

            color = color_map[param_value]

            # Plot mit gleicher Farbe für gleichen Parameter
            ax.plot(epoch, train_loss, alpha=0.5, color=color,
                    label=f'{param_name}={param_value}')
            ax.plot(epoch, val_loss, alpha=0.5, linestyle='--', color=color)

        # Entferne doppelte Labels in der Legende
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)

        ax.set_title(f'Loss by {param_name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    plt.savefig(output_file, dpi=300)


def heatmap(history, output_file='./figures/loss_history_heatmap.png'):
    data_list = []
    for data in history:
        loss = data[0]
        params = data[1]

        row = params.copy()
        row['final_train_loss'] = loss['train_loss'][-1]
        row['final_val_loss'] = loss['val_loss'][-1]
        row['min_val_loss'] = min(loss['val_loss'])
        data_list.append(row)

    df = pd.DataFrame(data_list)
    df['window_size'] = df['window_size'].fillna(0)  # None → 0

# NUR Parameter (X) vs Loss (Y)
    param_cols = ['q', 'd_model', 'n_latents',
                  'batch_size', 'n_heads', 'window_size']
    loss_cols = ['final_train_loss', 'final_val_loss', 'min_val_loss']

# Berechne nur Korrelation zwischen Parametern und Loss
    correlation = df[param_cols + loss_cols].corr().loc[param_cols, loss_cols]

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm',
                center=0, vmin=-1, vmax=1)
    plt.title('Parameter Influence on Loss')
    plt.savefig(output_file, dpi=300)


history = load_data()
plot_losses(history)
plot_loss_by_params(history)
heatmap(history)
