"""
10_run_extra_credit.py
Train and evaluate advanced GNN models for Extra Credit (+20%)

执行流程:
1. 加载已训练的baseline结果
2. 训练advanced models (RGCN, HGT, Hybrid)
3. 对比所有模型性能
4. 生成详细报告
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 确保能找到模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入Project 1模块
from graph_builder import build_ehrshot_graph

# 导入Project 2模块
from data_preparation import GNNDataPreparator
from gnn_models import create_model, HeteroToHomoWrapper
from train_evaluate import GNNTrainer
from torch_geometric.data import Data

# 导入Advanced models
from advanced_models import create_advanced_model, HeteroGNNTrainer

# 配置
DATA_PATH = "/home/henry/Desktop/LLM/GraphML/data/"
OUTPUT_DIR = "./project2_results"
BASELINE_DIR = "./project2_results"

# 超参数
HIDDEN_CHANNELS = 32
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200
PATIENCE = 50
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_baseline_results():
    """加载baseline结果"""
    print("\n[Loading] Baseline results...")
    
    baseline_file = os.path.join(BASELINE_DIR, 'model_comparison.csv')
    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)
        print("✓ Loaded baseline results:")
        print(baseline_df.to_string(index=False))
        return baseline_df
    else:
        print("⚠ Baseline results not found, will compare with new runs")
        return None


def train_advanced_models(preparator, hetero_data, train_mask, val_mask, test_mask):
    """训练所有advanced models"""
    
    print("\n" + "="*80)
    print("TRAINING ADVANCED GNN MODELS (Extra Credit +20%)")
    print("="*80)
    
    results_dict = {}
    
    # ==================== Model 1: R-GCN ====================
    print("\n[Advanced 1/3] Training R-GCN (Relational GCN)...")
    print("-" * 80)
    print("  Architecture: Relational convolution for heterogeneous graphs")
    print("  Paper: Schlichtkrull et al. (2018)")
    
    try:
        model_rgcn = create_advanced_model(
            'RGCN',
            hetero_data=hetero_data,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=4,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            num_bases=30
        )
        
        trainer_rgcn = HeteroGNNTrainer(model_rgcn, class_weights=preparator.class_weights)
        history_rgcn = trainer_rgcn.fit(
            hetero_data,
            train_mask,
            val_mask,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            verbose=True
        )
        
        results_rgcn = trainer_rgcn.test(hetero_data, test_mask, preparator.label_names)
        
        print(f"\n  R-GCN Results:")
        print(f"    Test Accuracy: {results_rgcn['test_acc']:.4f}")
        print(f"    Test F1 Score: {results_rgcn['test_f1']:.4f}")
        
        results_dict['R-GCN'] = results_rgcn
        
    except Exception as e:
        print(f"  ⚠ R-GCN training failed: {e}")
        results_dict['R-GCN'] = None
    
    # ==================== Model 2: HGT ====================
    print("\n[Advanced 2/3] Training HGT (Heterogeneous Graph Transformer)...")
    print("-" * 80)
    print("  Architecture: Attention mechanism for heterogeneous graphs")
    print("  Paper: Hu et al. (2020)")
    
    try:
        model_hgt = create_advanced_model(
            'HGT',
            hetero_data=hetero_data,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=4,
            num_heads=4,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
        
        trainer_hgt = HeteroGNNTrainer(model_hgt, class_weights=preparator.class_weights)
        history_hgt = trainer_hgt.fit(
            hetero_data,
            train_mask,
            val_mask,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            verbose=True
        )
        
        results_hgt = trainer_hgt.test(hetero_data, test_mask, preparator.label_names)
        
        print(f"\n  HGT Results:")
        print(f"    Test Accuracy: {results_hgt['test_acc']:.4f}")
        print(f"    Test F1 Score: {results_hgt['test_f1']:.4f}")
        
        results_dict['HGT'] = results_hgt
        
    except Exception as e:
        print(f"  ⚠ HGT training failed: {e}")
        results_dict['HGT'] = None
    
    # ==================== Model 3: Hybrid GAT-SAGE ====================
    print("\n[Advanced 3/3] Training Hybrid GAT-SAGE...")
    print("-" * 80)
    print("  Architecture: Custom hybrid combining GAT attention + SAGE efficiency")
    print("  Design: Alternating GAT/SAGE layers with skip connections")
    
    try:
        # 需要转换为同构图
        x, edge_index = HeteroToHomoWrapper.convert(hetero_data, target_node_type='drug')
        data = Data(x=x, edge_index=edge_index, y=hetero_data['drug'].y)
        
        model_hybrid = create_advanced_model(
            'HybridGATSAGE',
            in_channels=data.x.shape[1],
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=4,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            heads=4
        )
        
        trainer_hybrid = GNNTrainer(model_hybrid, class_weights=preparator.class_weights)
        history_hybrid = trainer_hybrid.fit(
            data,
            train_mask,
            val_mask,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            verbose=True
        )
        
        results_hybrid = trainer_hybrid.test(data, test_mask, preparator.label_names)
        
        print(f"\n  Hybrid GAT-SAGE Results:")
        print(f"    Test Accuracy: {results_hybrid['test_acc']:.4f}")
        print(f"    Test F1 Score: {results_hybrid['test_f1']:.4f}")
        
        results_dict['Hybrid-GATSAGE'] = results_hybrid
        
    except Exception as e:
        print(f"  ⚠ Hybrid training failed: {e}")
        results_dict['Hybrid-GATSAGE'] = None
    
    return results_dict


def compare_all_models(baseline_df, advanced_results):
    """对比所有模型"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    # 创建完整的对比表
    comparison_data = []
    
    # 添加baseline结果
    if baseline_df is not None:
        for _, row in baseline_df.iterrows():
            comparison_data.append({
                'Model': row['Model'],
                'Type': 'Baseline',
                'Test Acc': row['Test Acc'],
                'Test F1': row['Test F1'],
                'Test Loss': row['Test Loss']
            })
    
    # 添加advanced结果
    for model_name, results in advanced_results.items():
        if results is not None:
            comparison_data.append({
                'Model': model_name,
                'Type': 'Advanced (+20%)',
                'Test Acc': f"{results['test_acc']:.4f}",
                'Test F1': f"{results['test_f1']:.4f}",
                'Test Loss': f"{results['test_loss']:.4f}"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # 保存
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'full_model_comparison.csv'), index=False)
    print(f"\n✓ Saved to {OUTPUT_DIR}/full_model_comparison.csv")
    
    return comparison_df


def plot_comprehensive_comparison(comparison_df):
    """绘制完整对比图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 分离baseline和advanced
    models = comparison_df['Model'].values
    accs = [float(x) if isinstance(x, str) else x for x in comparison_df['Test Acc'].values]
    f1s = [float(x) if isinstance(x, str) else x for x in comparison_df['Test F1'].values]
    types = comparison_df['Type'].values
    
    # 颜色映射
    colors = ['steelblue' if t == 'Baseline' else 'coral' for t in types]
    
    # Accuracy对比
    bars1 = axes[0].bar(range(len(models)), accs, color=colors, alpha=0.8)
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylabel('Test Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14)
    axes[0].set_ylim([0.9, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)
    
    # F1 Score对比
    bars2 = axes[1].bar(range(len(models)), f1s, color=colors, alpha=0.8)
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_ylabel('Test F1 Score', fontsize=12)
    axes[1].set_title('Model F1 Score Comparison', fontsize=14)
    axes[1].set_ylim([0.9, 1.0])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Baseline Models'),
        Patch(facecolor='coral', label='Advanced Models (+20%)')
    ]
    axes[1].legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'comprehensive_model_comparison.png'),
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot")
    plt.close()


def generate_extra_credit_report(comparison_df, advanced_results):
    """生成Extra Credit报告"""
    
    report_path = os.path.join(OUTPUT_DIR, 'EXTRA_CREDIT_REPORT.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EXTRA CREDIT REPORT (+20%)\n")
        f.write("Advanced GNN Models for Drug Classification\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 实现的模型
        f.write("ADVANCED MODELS IMPLEMENTED:\n")
        f.write("-"*80 + "\n\n")
        
        f.write("1. R-GCN (Relational Graph Convolutional Network)\n")
        f.write("   - Paper: Schlichtkrull et al. (2018)\n")
        f.write("   - Key Feature: Handles multiple edge types in heterogeneous graphs\n")
        f.write("   - Architecture: Relational convolution with basis decomposition\n")
        f.write("   - Why relevant: Our EHR graph has multiple edge types (visit-drug, visit-disease)\n\n")
        
        f.write("2. HGT (Heterogeneous Graph Transformer)\n")
        f.write("   - Paper: Hu et al. (2020)\n")
        f.write("   - Key Feature: Attention mechanism for heterogeneous graphs\n")
        f.write("   - Architecture: Type-specific attention with multi-head design\n")
        f.write("   - Why relevant: Captures complex drug-disease relationships\n\n")
        
        f.write("3. Hybrid GAT-SAGE Model (Custom Design)\n")
        f.write("   - Innovation: Combines GAT's attention with SAGE's efficiency\n")
        f.write("   - Architecture: Alternating GAT/SAGE layers with skip connections\n")
        f.write("   - Why relevant: Balances expressiveness and computational efficiency\n\n")
        
        # 性能对比
        f.write("="*80 + "\n")
        f.write("PERFORMANCE COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # 最佳模型
        if len(advanced_results) > 0:
            valid_results = {k: v for k, v in advanced_results.items() if v is not None}
            if valid_results:
                best_advanced = max(valid_results.items(), key=lambda x: x[1]['test_acc'])
                
                f.write("BEST ADVANCED MODEL:\n")
                f.write("-"*80 + "\n")
                f.write(f"Model: {best_advanced[0]}\n")
                f.write(f"Test Accuracy: {best_advanced[1]['test_acc']:.4f}\n")
                f.write(f"Test F1 Score: {best_advanced[1]['test_f1']:.4f}\n\n")
        
        # 技术贡献
        f.write("="*80 + "\n")
        f.write("TECHNICAL CONTRIBUTIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. Heterogeneous Graph Modeling\n")
        f.write("   - Directly leverages multiple node and edge types\n")
        f.write("   - No information loss from homogeneous conversion\n\n")
        
        f.write("2. Advanced Attention Mechanisms\n")
        f.write("   - Type-aware attention in HGT\n")
        f.write("   - Multi-head attention for capturing diverse relationships\n\n")
        
        f.write("3. Custom Architecture Design\n")
        f.write("   - Hybrid model combines strengths of different architectures\n")
        f.write("   - Skip connections improve gradient flow\n\n")
        
        # 详细报告
        f.write("="*80 + "\n")
        f.write("DETAILED CLASSIFICATION REPORTS\n")
        f.write("="*80 + "\n\n")
        
        for model_name, results in advanced_results.items():
            if results is not None:
                f.write(f"{model_name}:\n")
                f.write("-"*80 + "\n")
                f.write(results['report'])
                f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Extra Credit report saved to {report_path}")


def main():
    """主执行函数"""
    
    print("="*80)
    print("EXTRA CREDIT: Advanced GNN Models (+20%)")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Step 1: 加载数据
    print("\n[STEP 1/4] Loading data...")
    graph_builder = build_ehrshot_graph(DATA_PATH)
    
    preparator = GNNDataPreparator(graph_builder)
    hetero_data, train_mask, val_mask, test_mask = preparator.prepare_full_data(feature_type='full')
    
    # Step 2: 加载baseline结果
    print("\n[STEP 2/4] Loading baseline results...")
    baseline_df = load_baseline_results()
    
    # Step 3: 训练advanced models
    print("\n[STEP 3/4] Training advanced models...")
    advanced_results = train_advanced_models(
        preparator, hetero_data, train_mask, val_mask, test_mask
    )
    
    # Step 4: 对比和报告
    print("\n[STEP 4/4] Generating comparison and reports...")
    comparison_df = compare_all_models(baseline_df, advanced_results)
    plot_comprehensive_comparison(comparison_df)
    generate_extra_credit_report(comparison_df, advanced_results)
    
    print("\n" + "="*80)
    print("✓ EXTRA CREDIT EVALUATION COMPLETE!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return comparison_df, advanced_results


if __name__ == "__main__":
    try:
        comparison_df, advanced_results = main()
        
        print("\n📊 Summary:")
        print("  ✓ Implemented 3 advanced models")
        print("  ✓ Compared with baseline models")
        print("  ✓ Generated comprehensive report")
        print("\n📁 Check project2_results/ for:")
        print("  - full_model_comparison.csv")
        print("  - EXTRA_CREDIT_REPORT.txt")
        print("  - figures/comprehensive_model_comparison.png")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
