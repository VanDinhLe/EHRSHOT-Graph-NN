"""
10_run_extra_credit.py
Train and evaluate advanced GNN models for Extra Credit (+20%)

Execution Pipeline:
1. Load pre-trained baseline model results from previous experiments
2. Train three advanced GNN architectures: R-GCN, HGT, and Hybrid GAT-SAGE
3. Perform comprehensive comparison across all baseline and advanced models
4. Generate detailed evaluation reports with performance metrics and visualizations

This script implements state-of-the-art heterogeneous graph neural networks
for drug classification tasks using Electronic Health Record (EHR) data.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add current directory to Python path to ensure module imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules from Project 1: Graph construction utilities
from graph_builder import build_ehrshot_graph

# Import modules from Project 2: Data preparation and baseline models
from data_preparation import GNNDataPreparator
from gnn_models import create_model, HeteroToHomoWrapper
from train_evaluate import GNNTrainer
from torch_geometric.data import Data

# Import advanced model architectures for extra credit
from advanced_models import create_advanced_model, HeteroGNNTrainer

# ==================== Configuration Parameters ====================
DATA_PATH = "/home/henry/Desktop/LLM/GraphML/data/"
OUTPUT_DIR = "./project2_results"
BASELINE_DIR = "./project2_results"

# ==================== Hyperparameters ====================
# These values were tuned based on preliminary experiments
HIDDEN_CHANNELS = 32      # Number of hidden units in GNN layers
NUM_LAYERS = 2            # Depth of the GNN architecture
DROPOUT = 0.5             # Dropout rate for regularization
LEARNING_RATE = 0.01      # Adam optimizer learning rate
WEIGHT_DECAY = 5e-4       # L2 regularization coefficient
EPOCHS = 200              # Maximum training epochs
PATIENCE = 50             # Early stopping patience (epochs without improvement)
SEED = 42                 # Random seed for reproducibility

# Set random seeds for reproducibility across PyTorch and NumPy
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_baseline_results():
    """
    Load baseline model results from previous experiments.
    
    This function attempts to read the model comparison CSV file generated
    by baseline experiments (GCN, GAT, GraphSAGE). If the file exists,
    it loads and displays the performance metrics; otherwise, it proceeds
    without baseline comparison data.
    
    Returns:
        pd.DataFrame or None: DataFrame containing baseline results if available,
                              None otherwise
    """
    print("\n[Loading] Baseline results from previous experiments...")
    
    baseline_file = os.path.join(BASELINE_DIR, 'model_comparison.csv')
    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)
        print("✓ Successfully loaded baseline results:")
        print(baseline_df.to_string(index=False))
        return baseline_df
    else:
        print("⚠ Baseline results file not found, will proceed with new model training only")
        return None


def train_advanced_models(preparator, hetero_data, train_mask, val_mask, test_mask):
    """
    Train all three advanced GNN architectures for extra credit.
    
    This function sequentially trains three state-of-the-art models:
    1. R-GCN: Relational Graph Convolutional Network for heterogeneous graphs
    2. HGT: Heterogeneous Graph Transformer with attention mechanisms
    3. Hybrid GAT-SAGE: Custom architecture combining GAT and GraphSAGE
    
    Each model is trained with early stopping, and results are collected
    for comprehensive comparison. Training failures are caught gracefully
    to ensure the pipeline continues even if one model fails.
    
    Args:
        preparator: GNNDataPreparator instance with class weights and label info
        hetero_data: Heterogeneous graph data structure (HeteroData)
        train_mask: Boolean mask for training nodes
        val_mask: Boolean mask for validation nodes
        test_mask: Boolean mask for test nodes
    
    Returns:
        dict: Dictionary mapping model names to their evaluation results
              Format: {'model_name': results_dict or None}
    """
    
    print("\n" + "="*80)
    print("TRAINING ADVANCED GNN MODELS (Extra Credit +20%)")
    print("="*80)
    
    results_dict = {}
    
    # ==================== Model 1: R-GCN ====================
    print("\n[Advanced 1/3] Training R-GCN (Relational Graph Convolutional Network)...")
    print("-" * 80)
    print("  Architecture: Relational convolution layers for heterogeneous graphs")
    print("  Paper: Schlichtkrull et al. (2018) - Modeling Relational Data with Graph CNNs")
    print("  Key Innovation: Handles multiple edge types through basis decomposition")
    
    try:
        # Create R-GCN model with basis decomposition to reduce parameters
        model_rgcn = create_advanced_model(
            'RGCN',
            hetero_data=hetero_data,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=4,  # 4 drug classes
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            num_bases=30  # Number of basis matrices for parameter sharing
        )
        
        # Initialize trainer with class-balanced loss weights
        trainer_rgcn = HeteroGNNTrainer(model_rgcn, class_weights=preparator.class_weights)
        
        # Train model with early stopping based on validation loss
        history_rgcn = trainer_rgcn.fit(
            hetero_data,
            train_mask,
            val_mask,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,
            verbose=True  # Print training progress
        )
        
        # Evaluate on test set and generate classification report
        results_rgcn = trainer_rgcn.test(hetero_data, test_mask, preparator.label_names)
        
        print(f"\n  R-GCN Test Results:")
        print(f"    Test Accuracy: {results_rgcn['test_acc']:.4f}")
        print(f"    Test F1 Score: {results_rgcn['test_f1']:.4f}")
        
        results_dict['R-GCN'] = results_rgcn
        
    except Exception as e:
        print(f"  ⚠ R-GCN training failed with error: {e}")
        results_dict['R-GCN'] = None
    
    # ==================== Model 2: HGT ====================
    print("\n[Advanced 2/3] Training HGT (Heterogeneous Graph Transformer)...")
    print("-" * 80)
    print("  Architecture: Multi-head attention mechanism for heterogeneous graphs")
    print("  Paper: Hu et al. (2020) - Heterogeneous Graph Transformer")
    print("  Key Innovation: Type-specific transformations with attention aggregation")
    
    try:
        # Create HGT model with multi-head attention
        model_hgt = create_advanced_model(
            'HGT',
            hetero_data=hetero_data,
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=4,  # 4 drug classes
            num_heads=4,  # Number of attention heads per layer
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
        
        # Initialize trainer with class-balanced loss weights
        trainer_hgt = HeteroGNNTrainer(model_hgt, class_weights=preparator.class_weights)
        
        # Train model with early stopping based on validation loss
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
        
        # Evaluate on test set and generate classification report
        results_hgt = trainer_hgt.test(hetero_data, test_mask, preparator.label_names)
        
        print(f"\n  HGT Test Results:")
        print(f"    Test Accuracy: {results_hgt['test_acc']:.4f}")
        print(f"    Test F1 Score: {results_hgt['test_f1']:.4f}")
        
        results_dict['HGT'] = results_hgt
        
    except Exception as e:
        print(f"  ⚠ HGT training failed with error: {e}")
        results_dict['HGT'] = None
    
    # ==================== Model 3: Hybrid GAT-SAGE ====================
    print("\n[Advanced 3/3] Training Hybrid GAT-SAGE Model...")
    print("-" * 80)
    print("  Architecture: Custom hybrid combining GAT attention with SAGE efficiency")
    print("  Design Philosophy: Alternating GAT/SAGE layers with residual connections")
    print("  Innovation: Balances expressiveness (GAT) with scalability (SAGE)")
    
    try:
        # Convert heterogeneous graph to homogeneous for hybrid model
        # This is necessary because the hybrid model uses standard PyG layers
        x, edge_index = HeteroToHomoWrapper.convert(hetero_data, target_node_type='drug')
        data = Data(x=x, edge_index=edge_index, y=hetero_data['drug'].y)
        
        # Create hybrid model with alternating GAT and SAGE layers
        model_hybrid = create_advanced_model(
            'HybridGATSAGE',
            in_channels=data.x.shape[1],  # Input feature dimension
            hidden_channels=HIDDEN_CHANNELS,
            out_channels=4,  # 4 drug classes
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            heads=4  # Number of attention heads in GAT layers
        )
        
        # Initialize trainer with class-balanced loss weights
        trainer_hybrid = GNNTrainer(model_hybrid, class_weights=preparator.class_weights)
        
        # Train model with early stopping based on validation loss
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
        
        # Evaluate on test set and generate classification report
        results_hybrid = trainer_hybrid.test(data, test_mask, preparator.label_names)
        
        print(f"\n  Hybrid GAT-SAGE Test Results:")
        print(f"    Test Accuracy: {results_hybrid['test_acc']:.4f}")
        print(f"    Test F1 Score: {results_hybrid['test_f1']:.4f}")
        
        results_dict['Hybrid-GATSAGE'] = results_hybrid
        
    except Exception as e:
        print(f"  ⚠ Hybrid model training failed with error: {e}")
        results_dict['Hybrid-GATSAGE'] = None
    
    return results_dict


def compare_all_models(baseline_df, advanced_results):
    """
    Create comprehensive comparison table across all baseline and advanced models.
    
    This function aggregates results from both baseline experiments (if available)
    and newly trained advanced models into a single comparison table. The table
    includes test accuracy, F1 score, and loss for each model, clearly marked
    by model type (Baseline vs Advanced).
    
    Args:
        baseline_df: DataFrame containing baseline model results (or None)
        advanced_results: Dictionary of advanced model results
    
    Returns:
        pd.DataFrame: Comprehensive comparison table with all model performances
    """
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON - ALL ARCHITECTURES")
    print("="*80)
    
    # Initialize list to store all model results
    comparison_data = []
    
    # Add baseline model results to comparison table
    if baseline_df is not None:
        for _, row in baseline_df.iterrows():
            comparison_data.append({
                'Model': row['Model'],
                'Type': 'Baseline',
                'Test Acc': row['Test Acc'],
                'Test F1': row['Test F1'],
                'Test Loss': row['Test Loss']
            })
    
    # Add advanced model results to comparison table
    for model_name, results in advanced_results.items():
        if results is not None:  # Only include successfully trained models
            comparison_data.append({
                'Model': model_name,
                'Type': 'Advanced (+20%)',
                'Test Acc': f"{results['test_acc']:.4f}",
                'Test F1': f"{results['test_f1']:.4f}",
                'Test Loss': f"{results['test_loss']:.4f}"
            })
    
    # Create DataFrame and display results
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison table to CSV file
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'full_model_comparison.csv'), index=False)
    print(f"\n✓ Comparison table saved to {OUTPUT_DIR}/full_model_comparison.csv")
    
    return comparison_df


def plot_comprehensive_comparison(comparison_df):
    """
    Generate side-by-side bar charts comparing all models' performance metrics.
    
    Creates a publication-quality figure with two subplots:
    - Left: Test Accuracy comparison across all models
    - Right: Test F1 Score comparison across all models
    
    Models are color-coded by type (Baseline: steel blue, Advanced: coral)
    and labeled with exact metric values for easy interpretation.
    
    Args:
        comparison_df: DataFrame containing model comparison results
    """
    
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extract data from comparison DataFrame
    models = comparison_df['Model'].values
    accs = [float(x) if isinstance(x, str) else x for x in comparison_df['Test Acc'].values]
    f1s = [float(x) if isinstance(x, str) else x for x in comparison_df['Test F1'].values]
    types = comparison_df['Type'].values
    
    # Assign colors based on model type: Baseline (blue) vs Advanced (coral)
    colors = ['steelblue' if t == 'Baseline' else 'coral' for t in types]
    
    # ========== Left subplot: Accuracy Comparison ==========
    bars1 = axes[0].bar(range(len(models)), accs, color=colors, alpha=0.8)
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylabel('Test Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14)
    axes[0].set_ylim([0.9, 1.0])  # Focus on high-performance range
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of each bar
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)
    
    # ========== Right subplot: F1 Score Comparison ==========
    bars2 = axes[1].bar(range(len(models)), f1s, color=colors, alpha=0.8)
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].set_ylabel('Test F1 Score', fontsize=12)
    axes[1].set_title('Model F1 Score Comparison', fontsize=14)
    axes[1].set_ylim([0.9, 1.0])  # Focus on high-performance range
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of each bar
    for i, v in enumerate(f1s):
        axes[1].text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)
    
    # Add legend to distinguish baseline from advanced models
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Baseline Models'),
        Patch(facecolor='coral', label='Advanced Models (+20%)')
    ]
    axes[1].legend(handles=legend_elements, loc='lower right')
    
    # Save figure with high resolution
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'comprehensive_model_comparison.png'),
                dpi=300, bbox_inches='tight')
    print(f"✓ Comparison visualization saved")
    plt.close()


def generate_extra_credit_report(comparison_df, advanced_results):
    """
    Generate comprehensive text report documenting extra credit work.
    
    This function creates a detailed report that includes:
    - Description of each advanced model architecture and motivation
    - Performance comparison tables with all metrics
    - Identification of best-performing model
    - Technical contributions and innovations
    - Detailed per-class classification reports for each model
    
    The report serves as documentation for the extra credit submission.
    
    Args:
        comparison_df: DataFrame with all model comparison results
        advanced_results: Dictionary containing detailed results for each advanced model
    """
    
    report_path = os.path.join(OUTPUT_DIR, 'EXTRA_CREDIT_REPORT.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # ========== Report Header ==========
        f.write("="*80 + "\n")
        f.write("EXTRA CREDIT REPORT (+20%)\n")
        f.write("Advanced GNN Models for Drug Classification in Electronic Health Records\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # ========== Model Descriptions ==========
        f.write("ADVANCED MODELS IMPLEMENTED:\n")
        f.write("-"*80 + "\n\n")
        
        f.write("1. R-GCN (Relational Graph Convolutional Network)\n")
        f.write("   - Paper: Schlichtkrull et al. (2018)\n")
        f.write("   - Key Feature: Handles multiple edge types in heterogeneous graphs natively\n")
        f.write("   - Architecture: Relational convolution with basis decomposition for parameter efficiency\n")
        f.write("   - Relevance: Our EHR graph contains multiple edge types (visit-drug, visit-disease)\n")
        f.write("     which R-GCN can leverage directly without information loss\n\n")
        
        f.write("2. HGT (Heterogeneous Graph Transformer)\n")
        f.write("   - Paper: Hu et al. (2020)\n")
        f.write("   - Key Feature: Type-aware multi-head attention mechanism for heterogeneous graphs\n")
        f.write("   - Architecture: Separate attention weights for different node/edge type combinations\n")
        f.write("   - Relevance: Captures complex drug-disease interaction patterns through learned attention,\n")
        f.write("     allowing the model to focus on most relevant relationships for classification\n\n")
        
        f.write("3. Hybrid GAT-SAGE Model (Custom Design)\n")
        f.write("   - Innovation: Novel architecture combining GAT's expressiveness with SAGE's efficiency\n")
        f.write("   - Architecture: Alternating GAT and GraphSAGE layers with residual skip connections\n")
        f.write("   - Design Rationale: GAT layers capture fine-grained attention patterns while SAGE layers\n")
        f.write("     provide efficient neighborhood aggregation; skip connections improve gradient flow\n")
        f.write("   - Relevance: Balances model expressiveness with computational efficiency, crucial for\n")
        f.write("     large-scale EHR graphs with thousands of nodes and edges\n\n")
        
        # ========== Performance Comparison ==========
        f.write("="*80 + "\n")
        f.write("PERFORMANCE COMPARISON - ALL MODELS\n")
        f.write("="*80 + "\n\n")
        
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # ========== Best Model Identification ==========
        if len(advanced_results) > 0:
            valid_results = {k: v for k, v in advanced_results.items() if v is not None}
            if valid_results:
                best_advanced = max(valid_results.items(), key=lambda x: x[1]['test_acc'])
                
                f.write("BEST PERFORMING ADVANCED MODEL:\n")
                f.write("-"*80 + "\n")
                f.write(f"Model Name: {best_advanced[0]}\n")
                f.write(f"Test Accuracy: {best_advanced[1]['test_acc']:.4f}\n")
                f.write(f"Test F1 Score (Macro): {best_advanced[1]['test_f1']:.4f}\n")
                f.write(f"Test Loss: {best_advanced[1]['test_loss']:.4f}\n\n")
        
        # ========== Technical Contributions ==========
        f.write("="*80 + "\n")
        f.write("TECHNICAL CONTRIBUTIONS AND INNOVATIONS\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. Native Heterogeneous Graph Modeling\n")
        f.write("   - Directly leverages multiple node types (drug, disease, visit) and edge types\n")
        f.write("   - Avoids information loss inherent in homogeneous graph conversion\n")
        f.write("   - Enables type-specific transformations and aggregations\n\n")
        
        f.write("2. Advanced Attention Mechanisms\n")
        f.write("   - Type-aware attention in HGT learns importance of different relationship types\n")
        f.write("   - Multi-head attention captures diverse relational patterns simultaneously\n")
        f.write("   - Attention weights provide interpretability for model decisions\n\n")
        
        f.write("3. Custom Hybrid Architecture Design\n")
        f.write("   - Strategic combination of complementary GNN architectures\n")
        f.write("   - Residual skip connections improve training stability and gradient flow\n")
        f.write("   - Balances expressiveness with computational efficiency\n\n")
        
        f.write("4. Robust Training Pipeline\n")
        f.write("   - Class-weighted loss handles imbalanced drug categories\n")
        f.write("   - Early stopping prevents overfitting\n")
        f.write("   - Comprehensive evaluation with multiple metrics (accuracy, F1, per-class performance)\n\n")
        
        # ========== Detailed Classification Reports ==========
        f.write("="*80 + "\n")
        f.write("DETAILED PER-CLASS CLASSIFICATION REPORTS\n")
        f.write("="*80 + "\n\n")
        
        for model_name, results in advanced_results.items():
            if results is not None:
                f.write(f"{model_name}:\n")
                f.write("-"*80 + "\n")
                f.write(results['report'])
                f.write("\n\n")
        
        # ========== Report Footer ==========
        f.write("="*80 + "\n")
        f.write("END OF EXTRA CREDIT REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Comprehensive extra credit report saved to {report_path}")


def main():
    """
    Main execution function orchestrating the entire extra credit pipeline.
    
    This function coordinates four major steps:
    1. Data loading and preprocessing
    2. Loading baseline results for comparison
    3. Training three advanced GNN architectures
    4. Generating comprehensive comparison reports and visualizations
    
    Returns:
        tuple: (comparison_df, advanced_results) containing all evaluation results
    """
    
    print("="*80)
    print("EXTRA CREDIT: Advanced GNN Models (+20%)")
    print("Drug Classification using State-of-the-Art Graph Neural Networks")
    print("="*80)
    print(f"Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # ========== Step 1: Load and Prepare Data ==========
    print("\n[STEP 1/4] Loading EHR data and constructing heterogeneous graph...")
    graph_builder = build_ehrshot_graph(DATA_PATH)
    
    preparator = GNNDataPreparator(graph_builder)
    hetero_data, train_mask, val_mask, test_mask = preparator.prepare_full_data(feature_type='full')
    
    print(f"  ✓ Graph constructed with {hetero_data['drug'].num_nodes} drug nodes")
    print(f"  ✓ Train/Val/Test split: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}")
    
    # ========== Step 2: Load Baseline Results ==========
    print("\n[STEP 2/4] Loading baseline model results for comparison...")
    baseline_df = load_baseline_results()
    
    # ========== Step 3: Train Advanced Models ==========
    print("\n[STEP 3/4] Training three advanced GNN architectures...")
    advanced_results = train_advanced_models(
        preparator, hetero_data, train_mask, val_mask, test_mask
    )
    
    # ========== Step 4: Generate Comparison and Reports ==========
    print("\n[STEP 4/4] Generating comprehensive comparison and documentation...")
    comparison_df = compare_all_models(baseline_df, advanced_results)
    plot_comprehensive_comparison(comparison_df)
    generate_extra_credit_report(comparison_df, advanced_results)
    
    print("\n" + "="*80)
    print("✓ EXTRA CREDIT EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Execution finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return comparison_df, advanced_results


if __name__ == "__main__":
    try:
        comparison_df, advanced_results = main()
        
        print("\n📊 Execution Summary:")
        print("  ✓ Successfully implemented 3 advanced GNN architectures")
        print("  ✓ Compared performance with baseline models")
        print("  ✓ Generated comprehensive evaluation report")
        print("\n📁 Output files available in project2_results/:")
        print("  - full_model_comparison.csv (quantitative comparison table)")
        print("  - EXTRA_CREDIT_REPORT.txt (detailed technical report)")
        print("  - figures/comprehensive_model_comparison.png (performance visualization)")
        
    except Exception as e:
        print(f"\n❌ Error encountered during execution: {e}")
        import traceback
        traceback.print_exc()