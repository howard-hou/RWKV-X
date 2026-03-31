import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.facecolor'] = '#f5f5f5'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1

output_dir = 'heatmaps_final_simple_uniform'
os.makedirs(output_dir, exist_ok=True)

csv_files = [
    'headwise/rwkv7-g1d-0.1b-20260129-ctx8192.headwise_memory_loss_stats.csv',
    'headwise/rwkv7-g1d-0.4b-20260210-ctx8192.headwise_memory_loss_stats.csv',
    'headwise/rwkv7-g1d-1.5b-20260212-ctx8192.headwise_memory_loss_stats.csv',
    'headwise/rwkv7-g1d-13.3b-20260131-ctx8192.headwise_memory_loss_stats.csv',
    'headwise/rwkv7-g1d-2.9b-20260131-ctx8192.headwise_memory_loss_stats.csv',
    'headwise/rwkv7-g1d-7.2b-20260131-ctx8192.headwise_memory_loss_stats.csv'
]

FIG_WIDTH = 20
FIG_HEIGHT = 15

TITLE_FONTSIZE = 32
LABEL_FONTSIZE = 28
TICK_FONTSIZE = 24
CBAR_LABEL_FONTSIZE = 26
CBAR_TICK_FONTSIZE = 22

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    
    pivot_table = df.pivot(index='head', columns='layer', values='mean')
    
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    model_name = base_name.replace('-', '_')
    
    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    cmap = sns.light_palette('#004d00', as_cmap=True, reverse=False)
    
    num_heads = len(pivot_table.index)
    num_layers = len(pivot_table.columns)
    
    max_heads = 40
    max_layers = 40
    
    scale_x = max_layers / num_layers if num_layers > 0 else 1
    scale_y = max_heads / num_heads if num_heads > 0 else 1
    
    ax = sns.heatmap(pivot_table, 
                   cmap=cmap,
                   annot=False, 
                   fmt='.6f',
                   cbar=True,
                   square=False,
                   linewidths=0.5,
                   linecolor='#e0e0e0',
                   vmin=pivot_table.min().min(),
                   vmax=pivot_table.max().max(),
                   cbar_kws={'shrink': 0.7, 'pad': 0.03, 'orientation': 'vertical'})
    
    import re
    match = re.search(r'g1d-(\d+\.\d+b)', base_name)
    if match:
        model_size = match.group(1)
        plt.title(f'Headwise Memory Loss Heatmap\nRWKV7-{model_size}', 
                 fontsize=TITLE_FONTSIZE, fontweight='bold', pad=30, loc='center')
        output_file = os.path.join(output_dir, f'rwkv7_g1d_{model_size}_heatmap.png')
    else:
        plt.title(f'Headwise Memory Loss Heatmap', 
                 fontsize=TITLE_FONTSIZE, fontweight='bold', pad=30, loc='center')
        output_file = os.path.join(output_dir, f'{model_name}_heatmap.png')
    plt.xlabel('Layer', fontsize=LABEL_FONTSIZE, fontweight='bold', labelpad=25)
    plt.ylabel('Head', fontsize=LABEL_FONTSIZE, fontweight='bold', labelpad=25)
    
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE, width=2, length=8, pad=15)
    
    plt.xticks(rotation=0)
    
    layers = sorted(df['layer'].unique())
    heads = sorted(df['head'].unique())
    
    if num_layers <= 10:
        layer_interval = 1
    elif num_layers <= 20:
        layer_interval = 2
    elif num_layers <= 30:
        layer_interval = 3
    else:
        layer_interval = max(1, num_layers // 10)
    
    if num_heads <= 10:
        head_interval = 1
    elif num_heads <= 20:
        head_interval = 2
    elif num_heads <= 30:
        head_interval = 3
    else:
        head_interval = max(1, num_heads // 10)
    
    ax.set_xticks(layers[::layer_interval])
    ax.set_xticklabels(layers[::layer_interval], fontweight='bold')
    
    ax.set_yticks(heads[::head_interval])
    ax.set_yticklabels(heads[::head_interval], fontweight='bold')
    
    ax.grid(True, color='#e0e0e0', linestyle='-', linewidth=0.5)
    
    cbar = ax.collections[0].colorbar
    cbar.set_label('Mean Memory Loss', fontsize=CBAR_LABEL_FONTSIZE, fontweight='bold', labelpad=20)
    cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE, pad=10)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f'Heatmap saved: {output_file}')

print('All heatmaps generated successfully!')