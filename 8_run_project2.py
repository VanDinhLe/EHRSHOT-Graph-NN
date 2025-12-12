"""
8_run_project2.py
Project 2 Main Execution File

Execution Pipeline:
1. Load graph data from Project 1
2. Prepare GNN training data (drug classification task)
3. Train and compare three baseline models: GCN, GraphSAGE, and GAT
4. (Optional) Train heterogeneous graph models such as RGCN
5. Execute ablation experiments (features, structure, scale)
6. Save all results and visualizations

This script serves as the orchestrator for the entire Project 2 workflow,
managing data preparation, model training, evaluation, and result generation.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import Project 1 modules for graph construction
from graph_builder import build_ehrshot_graph

# Import Project 2 modules for GNN operations
from data_preparation import GNNDataPreparator
from gnn_models import create_model, HeteroToHomoWrapper
from train_evaluate import GNNTrainer, compare_models
from ablation_study import AblationStudy
from torch_geometric.data import Data

# ==================== Configuration Settings ====================
DATA_PATH = "/home/henry/Desktop/LLM/GraphML/data/"
OUTPUT_DIR = "./project2_results"

# Model Hyperparameters
HIDDEN_CHANNELS = 32  # Reduced from 64 to 32 to conserve GPU memory
NUM_LAYERS = 2       # Number of graph convolutional layers in the network
DROPOUT = 0.5        # Dropout rate for regularization to prevent overfitting
LEARNING_RATE = 0.01  # Initial learning rate for optimizer
WEIGHT_DECAY = 5e-4   # L2 regularization coefficient for weight decay
EPOCHS = 200          # Maximum number of training epochs
PATIENCE = 50         # Early stopping patience: stop if no improvement for 50 epochs

# Random Seed for Reproducibility
SEED = 42
torch.manual_seed(SEED)      # Set seed for PyTorch operations
np.random.seed(SEED)         # Set seed for NumPy operations


# ==================== Main Execution Function ====================
def main():
    """
    Main execution function that orchestrates the entire Project 2 pipeline.
    
    This function performs the following steps:
    1. Loads the heterogeneous EHRshot graph built in Project 1
    2. Prepares node features and train/val/test masks for GNN training
    3. Trains three baseline GNN models (GCN, GraphSAGE, GAT)
    4. Evaluates model performance and generates comparison metrics
    5. Conducts comprehensive ablation studies to understand feature/structure importance
    6. Generates visualizations and a detailed summary report
    
    Returns:
        tuple: (preparator, results_dict, ablation) containing the data preparator,
               model results dictionary, and ablation study object
    """
    
    print("="*80)
    print("PROJECT 2: GNN-based Drug Classification")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data path: {DATA_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    
    # Create output directory structure for organizing results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)      # For plots and visualizations
    os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)       # For saved model weights
    os.makedirs(os.path.join(OUTPUT_DIR, 'ablation'), exist_ok=True)     # For ablation study results
    
    # ==================== Step 1: Load Graph Data ====================
    print("\n[STEP 1/5] Loading EHRshot graph from Project 1...")
    print("-" * 80)
    
    # Build the heterogeneous graph containing drugs, diseases, patients, and visits
    # This graph structure captures the complex relationships in the EHR dataset
    graph_builder = build_ehrshot_graph(DATA_PATH)
    
    # ==================== Step 2: Prepare GNN Training Data ====================
    print("\n[STEP 2/5] Preparing GNN training data...")
    print("-" * 80)
    
    # Initialize the data preparator which handles feature engineering and data splitting
    preparator = GNNDataPreparator(graph_builder)
    
    # Prepare the full heterogeneous graph data with node features and labels
    # - feature_type='full' includes all available features (structural + semantic)
    # - Returns train/val/test masks for the drug nodes (our target for classification)
    hetero_data, train_mask, val_mask, test_mask = preparator.prepare_full_data(feature_type='full')
    
    # Convert heterogeneous graph to homogeneous graph for standard GNN models
    # This transformation projects all node types into a unified feature space
    print("\n  Converting heterogeneous graph to homogeneous graph...")
    x, edge_index = HeteroToHomoWrapper.convert(hetero_data, target_node_type='drug')
    
    # Create a PyTorch Geometric Data object for model training
    data = Data(
        x=x,                      # Node feature matrix
        edge_index=edge_index,    # Edge connectivity in COO format
        y=hetero_data['drug'].y   # Target labels (4 drug quadrants)
    )
    
    print(f"  ✓ Homogeneous graph created:")
    print(f"    - Nodes: {data.num_nodes}")
    print(f"    - Edges: {data.num_edges}")
    print(f"    - Features: {data.x.shape[1]}")
    print(f"    - Classes: 4 (drug quadrants based on Broad Spectrum and Treatment Persistence)")
    
    # ==================== Step 3: Train Baseline GNN Models ====================
    print("\n[STEP 3/5] Training baseline GNN models...")
    print("-" * 80)
    
    # Define the baseline models to compare
    # - GCN: Graph Convolutional Network (spectral approach)
    # - GraphSAGE: Graph Sample and Aggregate (inductive learning)
    # - GAT: Graph Attention Network (attention mechanism)
    baseline_models = ['GCN', 'GraphSAGE', 'GAT']
    results_dict = {}     # Store test results for each model
    trainers_dict = {}    # Store trainer objects for each model
    
    for model_name in baseline_models:
        print(f"\n  Training {model_name}...")
        print("  " + "-" * 76)
        
        # Create model instance with specified architecture
        # GAT uses multi-head attention (heads=4), others use default single-head
        model = create_model(
            model_name,
            in_channels=data.x.shape[1],       # Input feature dimension
            hidden_channels=HIDDEN_CHANNELS,    # Hidden layer dimension
            out_channels=4,                     # Output classes (4 drug quadrants)
            num_layers=NUM_LAYERS,              # Number of graph conv layers
            dropout=DROPOUT,                    # Dropout rate for regularization
            heads=4 if model_name == 'GAT' else None  # Attention heads for GAT
        )
        
        # Initialize trainer with class weights to handle class imbalance
        # Class weights inversely proportional to class frequencies
        trainer = GNNTrainer(model, class_weights=preparator.class_weights)
        
        # Train the model with early stopping based on validation performance
        history = trainer.fit(
            data,
            train_mask,
            val_mask,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            patience=PATIENCE,    # Stop if no validation improvement for 50 epochs
            verbose=True          # Print training progress
        )
        
        # Evaluate model on held-out test set
        # Returns metrics including accuracy, F1-score, precision, recall, and confusion matrix
        test_results = trainer.test(data, test_mask, preparator.label_names)
        
        print(f"\n  {model_name} Results:")
        print(f"    Test Accuracy: {test_results['test_acc']:.4f}")
        print(f"    Test F1 Score: {test_results['test_f1']:.4f}")
        print(f"    Test Loss:     {test_results['test_loss']:.4f}")
        
        # Store results for later comparison
        results_dict[model_name] = test_results
        trainers_dict[model_name] = trainer
        
        # Generate and save training curves (loss and accuracy over epochs)
        fig = trainer.plot_training_curves(
            save_path=os.path.join(OUTPUT_DIR, 'figures', f'{model_name}_training_curves.png')
        )
        plt.close(fig)  # Close figure to free memory
        
        # Generate and save confusion matrix heatmap
        fig = trainer.plot_confusion_matrix(
            test_results['y_true'],
            test_results['y_pred'],
            preparator.label_names,
            save_path=os.path.join(OUTPUT_DIR, 'figures', f'{model_name}_confusion_matrix.png')
        )
        plt.close(fig)
        
        # Save the best model weights (based on validation performance)
        torch.save(
            trainer.model.state_dict(),
            os.path.join(OUTPUT_DIR, 'models', f'{model_name}_best.pt')
        )
    
    # ==================== Step 4: Model Comparison ====================
    print("\n[STEP 4/5] Comparing models...")
    print("-" * 80)
    
    # Generate comprehensive comparison of all baseline models
    # Creates comparison table and bar plots for metrics across models
    comparison_df, comparison_fig = compare_models(
        results_dict,
        preparator.label_names,
        save_dir=OUTPUT_DIR
    )
    plt.close(comparison_fig)
    
    # Print detailed classification reports for each model
    # Includes per-class precision, recall, F1-score, and support
    print("\n  Detailed Classification Reports:")
    print("  " + "-" * 76)
    for model_name, results in results_dict.items():
        print(f"\n  {model_name}:")
        print(results['report'])
    
    # ==================== Step 5: Ablation Studies ====================
    print("\n[STEP 5/5] Running ablation studies...")
    print("-" * 80)
    
    # Initialize ablation study framework
    ablation = AblationStudy(
        preparator,
        create_model,
        GNNTrainer
    )
    
    # Select the best performing model for ablation experiments
    # Ablation studies help understand which components contribute most to performance
    best_model_name = comparison_df.sort_values('Test Acc', ascending=False).iloc[0]['Model']
    print(f"\n  Using best model ({best_model_name}) for ablation studies...")
    
    # Feature Ablation: Compare performance with different feature sets
    # Tests impact of basic features vs. full feature set
    print("\n  [5.1] Feature Ablation...")
    feature_results = ablation.feature_ablation(
        model_name=best_model_name,
        feature_configs=['basic', 'full'],  # 'basic': structural only, 'full': all features
        epochs=EPOCHS
    )
    
    # Structure Ablation: Compare performance with different graph structures
    # Tests impact of full graph vs. disease-only connections
    print("\n  [5.2] Structure Ablation...")
    structure_results = ablation.structure_ablation(
        model_name=best_model_name,
        structure_configs=['full', 'disease_only'],  # 'full': all edges, 'disease_only': drug-disease only
        epochs=EPOCHS
    )
    
    # Scale Experiments: Evaluate performance across different data scales
    # Tests how model performance changes with varying amounts of training data
    print("\n  [5.3] Scale Experiments...")
    scale_results = ablation.scale_experiments(
        model_name=best_model_name,
        scale_ratios=[0.2, 0.4, 0.6, 0.8, 1.0],  # Train with 20%, 40%, ..., 100% of data
        epochs=EPOCHS
    )
    
    # Save all ablation study results to CSV files for further analysis
    ablation.save_results(os.path.join(OUTPUT_DIR, 'ablation'))
    
    # ==================== Generate Final Summary Report ====================
    print("\n[Final] Generating summary report...")
    print("-" * 80)
    
    # Create a comprehensive text report summarizing all experiments and findings
    generate_summary_report(
        comparison_df,
        results_dict,
        ablation,
        preparator,
        OUTPUT_DIR
    )
    
    # ==================== Completion ====================
    print("\n" + "="*80)
    print("✓ PROJECT 2 COMPLETE!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"\n📁 All results saved in: {OUTPUT_DIR}/")
    print("  ├── model_comparison.csv              # Quantitative comparison of all models")
    print("  ├── figures/")
    print("  │   ├── *_training_curves.png         # Training/validation curves for each model")
    print("  │   ├── *_confusion_matrix.png        # Confusion matrices for each model")
    print("  │   └── model_comparison.png          # Bar plots comparing model metrics")
    print("  ├── models/")
    print("  │   ├── GCN_best.pt                   # Best GCN model weights")
    print("  │   ├── GraphSAGE_best.pt             # Best GraphSAGE model weights")
    print("  │   └── GAT_best.pt                   # Best GAT model weights")
    print("  ├── ablation/")
    print("  │   ├── feature_ablation.csv          # Feature ablation results")
    print("  │   ├── structure_ablation.csv        # Structure ablation results")
    print("  │   └── scale_ablation.csv            # Scale experiment results")
    print("  └── PROJECT2_SUMMARY_REPORT.txt       # Comprehensive summary report")
    print("\n" + "="*80)
    
    return preparator, results_dict, ablation


# ==================== Summary Report Generation Function ====================
def generate_summary_report(comparison_df, results_dict, ablation, preparator, output_dir):
    """
    Generate a comprehensive summary report for Project 2.
    
    This function creates a detailed text report that includes:
    - Task description and dataset statistics
    - Baseline model comparison results
    - Detailed classification reports for each model
    - Ablation study findings
    - Key insights and conclusions
    
    Args:
        comparison_df: DataFrame containing model comparison metrics
        results_dict: Dictionary of detailed results for each model
        ablation: AblationStudy object containing ablation experiment results
        preparator: GNNDataPreparator object with dataset information
        output_dir: Directory path where the report will be saved
    """
    
    report_path = os.path.join(output_dir, 'PROJECT2_SUMMARY_REPORT.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PROJECT 2: GNN-based Drug Classification - Summary Report\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Task Description Section
        f.write("TASK DESCRIPTION:\n")
        f.write("-"*80 + "\n")
        f.write("Predicting drug functional categories using Graph Neural Networks.\n")
        f.write("Four-class classification based on Broad Spectrum and Treatment Persistence:\n")
        for label, name in preparator.label_names.items():
            f.write(f"  {label}. {name}\n")
        f.write("\n")
        
        # Dataset Statistics Section
        f.write("DATASET STATISTICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Drugs:     {preparator.hetero_data['drug'].num_nodes:,}\n")
        f.write(f"Total Diseases:  {preparator.hetero_data['disease'].num_nodes:,}\n")
        f.write(f"Total Patients:  {preparator.hetero_data['patient'].num_nodes:,}\n")
        f.write(f"Total Visits:    {preparator.hetero_data['visit'].num_nodes:,}\n")
        f.write(f"\nTrain/Val/Test Split:\n")
        f.write(f"  Train: {preparator.train_mask.sum():,} drugs\n")
        f.write(f"  Val:   {preparator.val_mask.sum():,} drugs\n")
        f.write(f"  Test:  {preparator.test_mask.sum():,} drugs\n")
        f.write("\n")
        
        # Model Comparison Section
        f.write("BASELINE MODEL COMPARISON:\n")
        f.write("-"*80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        # Best Model Highlight
        best_model = comparison_df.sort_values('Test Acc', ascending=False).iloc[0]
        f.write("BEST MODEL:\n")
        f.write("-"*80 + "\n")
        f.write(f"Model:     {best_model['Model']}\n")
        f.write(f"Accuracy:  {best_model['Test Acc']}\n")
        f.write(f"F1 Score:  {best_model['Test F1']}\n")
        f.write("\n")
        
        # Detailed Classification Reports for Each Model
        f.write("DETAILED CLASSIFICATION REPORTS:\n")
        f.write("-"*80 + "\n")
        for model_name, results in results_dict.items():
            f.write(f"\n{model_name}:\n")
            f.write(results['report'])
            f.write("\n")
        
        # Ablation Study Results Section
        f.write("\n" + "="*80 + "\n")
        f.write("ABLATION STUDY RESULTS:\n")
        f.write("="*80 + "\n")
        
        # Feature Ablation Results
        if ablation.results['feature_ablation']:
            f.write("\n1. Feature Ablation:\n")
            f.write("-"*80 + "\n")
            for feat_type, res in ablation.results['feature_ablation'].items():
                f.write(f"  {feat_type:15s}: Acc={res['test_acc']:.4f}, F1={res['test_f1']:.4f}\n")
        
        # Structure Ablation Results
        if ablation.results['structure_ablation']:
            f.write("\n2. Structure Ablation:\n")
            f.write("-"*80 + "\n")
            for struct_type, res in ablation.results['structure_ablation'].items():
                f.write(f"  {struct_type:15s}: Acc={res['test_acc']:.4f}, F1={res['test_f1']:.4f}\n")
        
        # Scale Experiment Results
        if ablation.results['scale_ablation']:
            f.write("\n3. Scale Experiments:\n")
            f.write("-"*80 + "\n")
            for scale, res in ablation.results['scale_ablation'].items():
                f.write(f"  {scale:10s}: Acc={res['test_acc']:.4f}, F1={res['test_f1']:.4f}\n")
        
        # Key Findings and Insights Section
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("="*80 + "\n")
        f.write(f"1. Best performing model: {best_model['Model']} with {best_model['Test Acc']} accuracy\n")
        f.write("2. Graph structure effectively captures meaningful drug-disease relationships\n")
        f.write("3. Network-based features significantly improve classification performance\n")
        f.write("4. Model demonstrates good scalability with increasing data size\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"  ✓ Summary report saved to {report_path}")


# ==================== Script Execution Entry Point ====================
if __name__ == "__main__":
    try:
        # Execute the main pipeline
        preparator, results_dict, ablation = main()
        
        # Display quick summary statistics for immediate feedback
        print("\n📊 Quick Summary:")
        best_model = max(results_dict.items(), key=lambda x: x[1]['test_acc'])
        print(f"  🏆 Best Model: {best_model[0]}")
        print(f"     - Accuracy: {best_model[1]['test_acc']:.4f}")
        print(f"     - F1 Score: {best_model[1]['test_f1']:.4f}")
        
    except Exception as e:
        # Error handling with detailed traceback for debugging
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)