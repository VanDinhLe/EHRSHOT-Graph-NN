"""
10_run_extra_credit_quick.py
Quick test of advanced models with reduced epochs
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from datetime import datetime

from graph_builder import build_ehrshot_graph
from data_preparation import GNNDataPreparator
from gnn_models import HeteroToHomoWrapper
from train_evaluate import GNNTrainer
from torch_geometric.data import Data
from advanced_models import create_advanced_model, HeteroGNNTrainer

# 配置
DATA_PATH = "/home/henry/Desktop/LLM/GraphML/data/"
HIDDEN_CHANNELS = 32
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 50  # 快速测试，只用50 epochs
PATIENCE = 20
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*80)
print("QUICK TEST: Advanced GNN Models")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 加载数据
print("\n[Step 1] Loading data...")
hetero_data = build_ehrshot_graph(DATA_PATH)

preparator = GNNDataPreparator(hetero_data)
hetero_data, train_mask, val_mask, test_mask = preparator.prepare_full_data(feature_type='full')

results = {}

# Test 1: R-GCN (Simplified)
print("\n[Test 1/3] R-GCN (Simplified)...")
try:
    model = create_advanced_model(
        'RGCN',
        hetero_data=hetero_data,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=4,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    trainer = HeteroGNNTrainer(model, class_weights=preparator.class_weights)
    trainer.fit(hetero_data, train_mask, val_mask, epochs=EPOCHS, patience=PATIENCE, verbose=True)
    result = trainer.test(hetero_data, test_mask, preparator.label_names)
    
    print(f"✓ R-GCN: Acc={result['test_acc']:.4f}, F1={result['test_f1']:.4f}")
    results['R-GCN'] = result
except Exception as e:
    print(f"✗ R-GCN failed: {e}")

# Test 2: HGT (Simplified)
print("\n[Test 2/3] HGT (Simplified)...")
try:
    model = create_advanced_model(
        'HGT',
        hetero_data=hetero_data,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=4,
        num_heads=4,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    trainer = HeteroGNNTrainer(model, class_weights=preparator.class_weights)
    trainer.fit(hetero_data, train_mask, val_mask, epochs=EPOCHS, patience=PATIENCE, verbose=True)
    result = trainer.test(hetero_data, test_mask, preparator.label_names)
    
    print(f"✓ HGT: Acc={result['test_acc']:.4f}, F1={result['test_f1']:.4f}")
    results['HGT'] = result
except Exception as e:
    print(f"✗ HGT failed: {e}")

# Test 3: Hybrid GAT-SAGE
print("\n[Test 3/3] Hybrid GAT-SAGE...")
try:
    x, edge_index = HeteroToHomoWrapper.convert(hetero_data, target_node_type='drug')
    data = Data(x=x, edge_index=edge_index, y=hetero_data['drug'].y)
    
    model = create_advanced_model(
        'HybridGATSAGE',
        in_channels=data.x.shape[1],
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=4,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        heads=4
    )
    
    trainer = GNNTrainer(model, class_weights=preparator.class_weights)
    trainer.fit(data, train_mask, val_mask, epochs=EPOCHS, patience=PATIENCE, verbose=True)
    result = trainer.test(data, test_mask, preparator.label_names)
    
    print(f"✓ Hybrid: Acc={result['test_acc']:.4f}, F1={result['test_f1']:.4f}")
    results['Hybrid'] = result
except Exception as e:
    print(f"✗ Hybrid failed: {e}")

# Summary
print("\n" + "="*80)
print("QUICK TEST RESULTS")
print("="*80)
for model_name, result in results.items():
    print(f"{model_name:20s}: Acc={result['test_acc']:.4f}, F1={result['test_f1']:.4f}")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nIf tests passed, run: python 10_run_extra_credit.py")
