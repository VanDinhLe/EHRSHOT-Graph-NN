"""
7_ablation_study.py
Ablation Study Module

This module contains comprehensive ablation experiments including:
1. Feature Ablation: Analyzes the impact of different feature combinations on model performance
2. Graph Structure Ablation: Evaluates how different graph structures affect model accuracy
3. Scale Experiments: Investigates model performance across various dataset sizes
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import os


class AblationStudy:
    """
    Ablation Study Manager
    
    This class manages and orchestrates all ablation experiments for GNN models.
    It provides systematic methods to evaluate the contribution of different components
    to the overall model performance.
    """
    
    def __init__(self, preparator, model_creator_func, trainer_class):
        """
        Initialize the Ablation Study Manager
        
        Args:
            preparator: GNNDataPreparator instance that handles data preparation and preprocessing
            model_creator_func: Function that creates and returns a model instance
            trainer_class: Trainer class used for model training and evaluation
        """
        self.preparator = preparator
        self.model_creator_func = model_creator_func
        self.trainer_class = trainer_class
        
        # Dictionary to store results from all ablation experiments
        self.results = {
            'feature_ablation': {},      # Results from feature ablation experiments
            'structure_ablation': {},    # Results from graph structure ablation experiments
            'scale_ablation': {}         # Results from scale experiments
        }
    
    # ==================== 1. Feature Ablation Studies ====================
    def feature_ablation(self, model_name='GCN', feature_configs=None, 
                        epochs=200, seed=42):
        """
        Conduct Feature Ablation Experiments
        
        This method systematically removes or modifies different feature sets to evaluate
        their individual contributions to model performance. It helps identify which features
        are most critical for accurate predictions.
        
        Args:
            model_name: Name of the model to use (e.g., 'GCN', 'GAT', 'GraphSAGE')
            feature_configs: List of feature configuration types to test
                            Examples: ['basic', 'full', 'disease_only', 'symptom_only']
            epochs: Number of training epochs for each configuration
            seed: Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing test accuracy, F1 score, and loss for each feature config
        """
        print("\n" + "="*80)
        print("FEATURE ABLATION STUDY")
        print("="*80)
        
        # Use default feature configurations if none provided
        if feature_configs is None:
            feature_configs = ['basic', 'full']
        
        results = {}
        
        # Iterate through each feature configuration
        for feat_type in feature_configs:
            print(f"\n[Feature Ablation] Testing with features: {feat_type}")
            
            # Prepare data with specific feature type
            self.preparator.prepare_full_data(feature_type=feat_type)
            hetero_data = self.preparator.hetero_data
            
            # Convert heterogeneous graph to homogeneous graph for standard GNN models
            # This is necessary for GCN/GAT/GraphSAGE which expect homogeneous input
            from gnn_models import HeteroToHomoWrapper
            x, edge_index = HeteroToHomoWrapper.convert(hetero_data, target_node_type='drug')
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=x,                      # Node feature matrix
                edge_index=edge_index,    # Graph connectivity in COO format
                y=hetero_data['drug'].y   # Target labels
            )
            
            # Create model with appropriate input dimension
            in_channels = data.x.shape[1]
            model = self.model_creator_func(
                model_name, 
                in_channels=in_channels,     # Input feature dimension
                hidden_channels=64,          # Hidden layer dimension
                out_channels=4               # Number of output classes
            )
            
            # Train the model
            torch.manual_seed(seed)  # Ensure reproducibility
            trainer = self.trainer_class(model, class_weights=self.preparator.class_weights)
            trainer.fit(
                data, 
                self.preparator.train_mask,   # Training set mask
                self.preparator.val_mask,     # Validation set mask
                epochs=epochs,
                verbose=False                 # Suppress detailed training logs
            )
            
            # Evaluate on test set
            test_results = trainer.test(
                data, 
                self.preparator.test_mask,    # Test set mask
                self.preparator.label_names   # Class labels for reporting
            )
            
            # Store results for this feature configuration
            results[feat_type] = {
                'test_acc': test_results['test_acc'],
                'test_f1': test_results['test_f1'],
                'test_loss': test_results['test_loss']
            }
            
            print(f"  ✓ Test Accuracy: {test_results['test_acc']:.4f}")
            print(f"  ✓ Test F1 Score: {test_results['test_f1']:.4f}")
        
        # Save results internally
        self.results['feature_ablation'] = results
        
        # Visualize comparison across different feature configurations
        self._plot_ablation_results(results, 'Feature Type', 'Feature Ablation Study')
        
        return results
    
    # ==================== 2. Graph Structure Ablation Studies ====================
    def structure_ablation(self, model_name='GCN', structure_configs=None, 
                          epochs=200, seed=42):
        """
        Conduct Graph Structure Ablation Experiments
        
        This method evaluates the importance of different edge types and graph structures
        by systematically removing or modifying specific connections in the graph.
        
        Args:
            model_name: Name of the GNN model to use
            structure_configs: List of graph structure configurations to test
                Examples:
                - 'full': Use complete graph with all edge types
                - 'disease_only': Only use drug-visit-disease edges
                - 'patient_only': Only use drug-visit-patient edges
                - 'no_visit': Create direct drug-disease connections, bypassing visit nodes
            epochs: Number of training epochs
            seed: Random seed for reproducibility
            
        Returns:
            dict: Results including accuracy, F1, loss, and number of edges for each config
        """
        print("\n" + "="*80)
        print("GRAPH STRUCTURE ABLATION STUDY")
        print("="*80)
        
        # Use default structure configurations if none provided
        if structure_configs is None:
            structure_configs = ['full', 'disease_only', 'no_visit']
        
        results = {}
        
        # Prepare the base data with full features
        self.preparator.prepare_full_data(feature_type='full')
        hetero_data = self.preparator.hetero_data
        
        # Test each graph structure configuration
        for struct_type in structure_configs:
            print(f"\n[Structure Ablation] Testing with structure: {struct_type}")
            
            # Modify graph structure based on configuration type
            if struct_type == 'full':
                # Use complete graph with all edge types
                modified_hetero = hetero_data
            
            elif struct_type == 'disease_only':
                # Retain only drug-visit-disease pathway
                # This isolates the contribution of disease-related connections
                modified_hetero = deepcopy(hetero_data)
                # Remove other edge types by setting them to empty tensors
                modified_hetero['visit', 'belongs_to', 'patient'].edge_index = torch.tensor([[], []], dtype=torch.long)
                modified_hetero['visit', 'has_symptom', 'symptom'].edge_index = torch.tensor([[], []], dtype=torch.long)
            
            elif struct_type == 'no_visit':
                # Build direct drug-disease graph, bypassing visit nodes
                # This tests if intermediate visit nodes add value
                modified_hetero = self._build_drug_disease_graph(hetero_data)
            
            # Convert modified heterogeneous graph to homogeneous format
            from gnn_models import HeteroToHomoWrapper
            x, edge_index = HeteroToHomoWrapper.convert(modified_hetero, target_node_type='drug')
            
            # Create data object with modified structure
            data = Data(
                x=x,
                edge_index=edge_index,
                y=hetero_data['drug'].y
            )
            
            # Initialize and train model
            in_channels = data.x.shape[1]
            model = self.model_creator_func(
                model_name,
                in_channels=in_channels,
                hidden_channels=64,
                out_channels=4
            )
            
            torch.manual_seed(seed)
            trainer = self.trainer_class(model, class_weights=self.preparator.class_weights)
            trainer.fit(
                data,
                self.preparator.train_mask,
                self.preparator.val_mask,
                epochs=epochs,
                verbose=False
            )
            
            # Evaluate on test set
            test_results = trainer.test(
                data,
                self.preparator.test_mask,
                self.preparator.label_names
            )
            
            # Store results including graph statistics
            results[struct_type] = {
                'test_acc': test_results['test_acc'],
                'test_f1': test_results['test_f1'],
                'test_loss': test_results['test_loss'],
                'num_edges': edge_index.shape[1]  # Track number of edges for reference
            }
            
            print(f"  ✓ Number of edges: {edge_index.shape[1]}")
            print(f"  ✓ Test Accuracy:   {test_results['test_acc']:.4f}")
            print(f"  ✓ Test F1 Score:   {test_results['test_f1']:.4f}")
        
        # Save results
        self.results['structure_ablation'] = results
        
        # Visualize comparison across different graph structures
        self._plot_ablation_results(results, 'Graph Structure', 'Structure Ablation Study')
        
        return results
    
    def _build_drug_disease_graph(self, hetero_data):
        """
        Build Direct Drug-Disease Graph (Bypassing Visit Nodes)
        
        This helper method creates direct edges between drugs and diseases by
        removing the intermediate visit nodes. This helps evaluate whether the
        hierarchical structure adds value or if direct connections suffice.
        
        Args:
            hetero_data: Original heterogeneous graph data
            
        Returns:
            Modified heterogeneous graph with direct drug-disease connections
            
        Note:
            Actual implementation would be more complex, involving:
            - Identifying all drugs connected to diseases through visits
            - Creating direct edges between them
            - Potentially aggregating or averaging intermediate features
        """
        modified = deepcopy(hetero_data)
        # Simplified placeholder - actual implementation would require
        # traversing drug->visit->disease paths and creating direct edges
        return modified
    
    # ==================== 3. Scale Experiments ====================
    def scale_experiments(self, model_name='GCN', scale_ratios=None, 
                         epochs=200, seed=42):
        """
        Conduct Scale Experiments
        
        This method evaluates model performance across different dataset sizes to understand:
        - How much data is needed for good performance
        - Whether the model scales well with more data
        - If there are diminishing returns at larger scales
        
        Args:
            model_name: Name of the GNN model to use
            scale_ratios: List of data size ratios to test (e.g., [0.1, 0.25, 0.5, 0.75, 1.0])
                         Each ratio represents the proportion of full dataset to use
            epochs: Number of training epochs for each scale
            seed: Random seed for reproducibility
            
        Returns:
            dict: Results including accuracy, F1, and graph statistics for each scale
        """
        print("\n" + "="*80)
        print("SCALE EXPERIMENTS")
        print("="*80)
        
        # Use default scale ratios if none provided
        if scale_ratios is None:
            scale_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        results = {}
        
        # Prepare full dataset
        self.preparator.prepare_full_data(feature_type='full')
        hetero_data = self.preparator.hetero_data
        
        # Convert to homogeneous graph
        from gnn_models import HeteroToHomoWrapper
        x, edge_index = HeteroToHomoWrapper.convert(hetero_data, target_node_type='drug')
        
        # Create full data object
        full_data = Data(
            x=x,
            edge_index=edge_index,
            y=hetero_data['drug'].y
        )
        
        # Test at each scale ratio
        for ratio in scale_ratios:
            print(f"\n[Scale Experiment] Training with {ratio*100:.0f}% of data")
            
            # Create subgraph if using less than full data
            if ratio < 1.0:
                # Randomly sample nodes proportional to desired ratio
                num_nodes = int(full_data.num_nodes * ratio)
                torch.manual_seed(seed)
                sampled_nodes = torch.randperm(full_data.num_nodes)[:num_nodes]
                
                # Extract subgraph containing only sampled nodes
                # This also updates edge indices to maintain connectivity
                sub_edge_index, _ = subgraph(
                    sampled_nodes, 
                    full_data.edge_index,
                    relabel_nodes=True  # Relabel nodes to be contiguous [0, num_sampled_nodes)
                )
                
                # Create corresponding train/val/test masks for subgraph
                train_mask_sub = self.preparator.train_mask[sampled_nodes]
                val_mask_sub = self.preparator.val_mask[sampled_nodes]
                test_mask_sub = self.preparator.test_mask[sampled_nodes]
                
                # Create subgraph data object
                data = Data(
                    x=full_data.x[sampled_nodes],
                    edge_index=sub_edge_index,
                    y=full_data.y[sampled_nodes]
                )
            else:
                # Use full dataset
                data = full_data
                train_mask_sub = self.preparator.train_mask
                val_mask_sub = self.preparator.val_mask
                test_mask_sub = self.preparator.test_mask
            
            # Initialize and train model
            in_channels = data.x.shape[1]
            model = self.model_creator_func(
                model_name,
                in_channels=in_channels,
                hidden_channels=64,
                out_channels=4
            )
            
            torch.manual_seed(seed)
            trainer = self.trainer_class(model, class_weights=self.preparator.class_weights)
            trainer.fit(
                data,
                train_mask_sub,
                val_mask_sub,
                epochs=epochs,
                verbose=False
            )
            
            # Evaluate on test set
            test_results = trainer.test(
                data,
                test_mask_sub,
                self.preparator.label_names
            )
            
            # Store results with graph statistics
            results[f"{ratio*100:.0f}%"] = {
                'test_acc': test_results['test_acc'],
                'test_f1': test_results['test_f1'],
                'num_nodes': data.num_nodes,      # Track number of nodes at this scale
                'num_edges': data.edge_index.shape[1]  # Track number of edges at this scale
            }
            
            print(f"  ✓ Number of nodes: {data.num_nodes}")
            print(f"  ✓ Test Accuracy:   {test_results['test_acc']:.4f}")
            print(f"  ✓ Test F1 Score:   {test_results['test_f1']:.4f}")
        
        # Save results
        self.results['scale_ablation'] = results
        
        # Visualize performance curves across different scales
        self._plot_scale_results(results, scale_ratios)
        
        return results
    
    # ==================== Visualization Methods ====================
    def _plot_ablation_results(self, results, xlabel, title):
        """
        Plot Ablation Experiment Results
        
        Creates a comprehensive visualization with three subplots showing:
        1. Test accuracy comparison
        2. Test F1 score comparison
        3. Test loss comparison
        
        Args:
            results: Dictionary of results from ablation experiments
            xlabel: Label for x-axis (e.g., 'Feature Type', 'Graph Structure')
            title: Title prefix for the plots
            
        Returns:
            matplotlib.figure.Figure: The created figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Extract data from results
        configs = list(results.keys())
        accs = [results[c]['test_acc'] for c in configs]
        f1s = [results[c]['test_f1'] for c in configs]
        losses = [results[c]['test_loss'] for c in configs]
        
        # Plot 1: Accuracy Comparison
        axes[0].bar(configs, accs, color='steelblue', alpha=0.8)
        axes[0].set_ylabel('Test Accuracy', fontsize=12)
        axes[0].set_xlabel(xlabel, fontsize=12)
        axes[0].set_title(f'{title} - Accuracy', fontsize=14)
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3, axis='y')
        # Add value labels on bars
        for i, v in enumerate(accs):
            axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        # Plot 2: F1 Score Comparison
        axes[1].bar(configs, f1s, color='coral', alpha=0.8)
        axes[1].set_ylabel('Test F1 Score', fontsize=12)
        axes[1].set_xlabel(xlabel, fontsize=12)
        axes[1].set_title(f'{title} - F1 Score', fontsize=14)
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='y')
        # Add value labels on bars
        for i, v in enumerate(f1s):
            axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        # Plot 3: Loss Comparison
        axes[2].bar(configs, losses, color='mediumseagreen', alpha=0.8)
        axes[2].set_ylabel('Test Loss', fontsize=12)
        axes[2].set_xlabel(xlabel, fontsize=12)
        axes[2].set_title(f'{title} - Loss', fontsize=14)
        axes[2].grid(True, alpha=0.3, axis='y')
        # Add value labels on bars
        for i, v in enumerate(losses):
            axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _plot_scale_results(self, results, scale_ratios):
        """
        Plot Scale Experiment Results
        
        Creates learning curves showing how model performance changes with dataset size.
        This visualization helps identify:
        - Minimum data requirements for acceptable performance
        - Whether performance saturates at larger scales
        - Data efficiency of the model
        
        Args:
            results: Dictionary of results from scale experiments
            scale_ratios: List of scale ratios that were tested
            
        Returns:
            matplotlib.figure.Figure: The created figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Extract data for plotting
        configs = [f"{r*100:.0f}%" for r in scale_ratios]
        accs = [results[c]['test_acc'] for c in configs]
        f1s = [results[c]['test_f1'] for c in configs]
        
        # Plot 1: Accuracy vs Scale
        axes[0].plot(scale_ratios, accs, marker='o', linewidth=2, markersize=8, color='steelblue')
        axes[0].set_xlabel('Data Scale Ratio', fontsize=12)
        axes[0].set_ylabel('Test Accuracy', fontsize=12)
        axes[0].set_title('Model Performance vs Data Scale', fontsize=14)
        axes[0].set_ylim([0, 1])
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: F1 Score vs Scale
        axes[1].plot(scale_ratios, f1s, marker='s', linewidth=2, markersize=8, color='coral')
        axes[1].set_xlabel('Data Scale Ratio', fontsize=12)
        axes[1].set_ylabel('Test F1 Score', fontsize=12)
        axes[1].set_title('Model F1 Score vs Data Scale', fontsize=14)
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def save_results(self, save_dir):
        """
        Save All Ablation Study Results to CSV Files
        
        Exports results from all experiments (feature, structure, and scale ablations)
        to separate CSV files for further analysis and reporting.
        
        Args:
            save_dir: Directory path where result files will be saved
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each experiment type's results
        for exp_name, results in self.results.items():
            if results:  # Only save if results exist
                # Convert results dictionary to DataFrame for easy CSV export
                df = pd.DataFrame(results).T
                # Save to CSV with experiment name
                df.to_csv(os.path.join(save_dir, f'{exp_name}.csv'))
                print(f"  ✓ Saved {exp_name} results to CSV")


# ==================== Testing and Demonstration ====================
if __name__ == "__main__":
    print("This module should be imported and used with actual data")
    print("Example usage:")
    print("  from ablation_study import AblationStudy")
    print("  ablation = AblationStudy(preparator, model_creator, trainer_class)")
    print("  ablation.feature_ablation(model_name='GCN')")
    print("  ablation.structure_ablation(model_name='GAT')")
    print("  ablation.scale_experiments(model_name='GraphSAGE')")