"""
9_advanced_models.py
Advanced GNN Models for Extra Credit (+20%)

This module implements three advanced Graph Neural Network architectures:
1. R-GCN (Relational Graph Convolutional Network) - Designed for heterogeneous graphs
2. HGT (Heterogeneous Graph Transformer) - Attention-based heterogeneous graph model
3. Hybrid GAT-SAGE Model - Custom architecture combining GAT attention with SAGE efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, HGTConv, Linear, to_hetero
from torch_geometric.nn import GATConv, SAGEConv


# ==================== 1. R-GCN for Heterogeneous Graph ====================
class HeteroRGCN(nn.Module):
    """
    Simplified RGCN-inspired model for heterogeneous graphs
    
    This implementation uses metapath aggregation instead of full RGCN for improved
    memory efficiency while maintaining the ability to handle multiple node and edge types.
    
    Reference Paper: 
    Schlichtkrull et al. (2018) "Modeling Relational Data with Graph Convolutional Networks"
    
    Architecture:
    - Input projection for drug node features
    - Multiple linear transformation layers with batch normalization
    - ReLU activation and dropout for regularization
    - Final linear layer for classification
    
    Args:
        hetero_data: Heterogeneous graph data object containing node features and edge indices
        hidden_channels: Dimension of hidden representations (default: 32)
        out_channels: Number of output classes (default: 4)
        num_layers: Number of message passing layers (default: 2)
        dropout: Dropout probability for regularization (default: 0.5)
        num_bases: Number of basis matrices for relation-specific transformations (default: 30)
    """
    def __init__(self, hetero_data, hidden_channels=32, out_channels=4, 
                 num_layers=2, dropout=0.5, num_bases=30):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Extract input feature dimensions from drug nodes
        drug_in_channels = hetero_data['drug'].x.shape[1]
        
        # Initial projection layer to transform drug features into hidden space
        self.drug_proj = Linear(drug_in_channels, hidden_channels)
        
        # Create message passing layers with batch normalization
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Linear transformation for each layer
            self.convs.append(Linear(hidden_channels, hidden_channels))
            # Batch normalization for stable training
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Final output projection layer for classification
        self.lin_out = Linear(hidden_channels, out_channels)
        
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through the network
        
        Args:
            x_dict: Dictionary mapping node types to feature tensors
            edge_index_dict: Dictionary mapping edge types to edge indices
            
        Returns:
            Output logits for classification [num_nodes, out_channels]
        """
        # Project drug node features to hidden space
        x = self.drug_proj(x_dict['drug'])
        
        # Apply sequential transformation layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x)  # Linear transformation
            x = norm(x)  # Normalize features
            x = F.relu(x)  # Non-linear activation
            x = F.dropout(x, p=self.dropout, training=self.training)  # Regularization
        
        # Generate final output logits
        out = self.lin_out(x)
        return out


# ==================== 2. HGT (Heterogeneous Graph Transformer) ====================
class HGTModel(nn.Module):
    """
    Simplified Heterogeneous Graph Transformer with Multi-Head Attention
    
    This model adapts the transformer architecture for heterogeneous graphs,
    using multi-head self-attention to capture complex relationships between nodes.
    Simplified for improved stability and memory efficiency compared to full HGT.
    
    Reference Paper:
    Hu et al. (2020) "Heterogeneous Graph Transformer"
    
    Architecture:
    - Input projection to hidden space
    - Multiple transformer layers with:
        * Multi-head self-attention mechanism
        * Layer normalization with residual connections
        * Position-wise feed-forward networks (FFN)
    - Final linear projection for classification
    
    Args:
        hetero_data: Heterogeneous graph data object
        hidden_channels: Dimension of hidden representations (default: 32)
        out_channels: Number of output classes (default: 4)
        num_heads: Number of attention heads (default: 4)
        num_layers: Number of transformer layers (default: 2)
        dropout: Dropout probability for regularization (default: 0.5)
    """
    def __init__(self, hetero_data, hidden_channels=32, out_channels=4,
                 num_heads=4, num_layers=2, dropout=0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        
        # Extract input feature dimensions from drug nodes
        drug_in_channels = hetero_data['drug'].x.shape[1]
        
        # Project input features to hidden dimension
        self.lin_in = Linear(drug_in_channels, hidden_channels)
        
        # Multi-head attention layers for capturing node relationships
        self.attns = nn.ModuleList()
        # Layer normalization for each attention layer
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # Multi-head attention with dropout
            self.attns.append(nn.MultiheadAttention(
                hidden_channels, 
                num_heads, 
                dropout=dropout,
                batch_first=True  # Input format: [batch, sequence, features]
            ))
            # Layer normalization for stable training
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Position-wise feed-forward networks (FFN) for each layer
        self.ffns = nn.ModuleList()
        for _ in range(num_layers):
            self.ffns.append(nn.Sequential(
                Linear(hidden_channels, hidden_channels * 2),  # Expand
                nn.ReLU(),  # Non-linearity
                nn.Dropout(dropout),  # Regularization
                Linear(hidden_channels * 2, hidden_channels)  # Project back
            ))
        
        # Final output projection layer
        self.lin_out = Linear(hidden_channels, out_channels)
        
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through the heterogeneous graph transformer
        
        Args:
            x_dict: Dictionary mapping node types to feature tensors
            edge_index_dict: Dictionary mapping edge types to edge indices
            
        Returns:
            Output logits for classification [num_nodes, out_channels]
        """
        # Project drug node features to hidden space
        x = self.lin_in(x_dict['drug'])
        
        # Add sequence dimension for attention mechanism [N, D] -> [N, 1, D]
        x = x.unsqueeze(1)
        
        # Apply transformer layers sequentially
        for attn, norm, ffn in zip(self.attns, self.norms, self.ffns):
            # Multi-head self-attention with residual connection
            attn_out, _ = attn(x, x, x)  # Query, Key, Value are all x
            x = norm(x + attn_out)  # Add & Norm
            
            # Feed-forward network with residual connection
            ffn_out = ffn(x)
            x = norm(x + ffn_out)  # Add & Norm
        
        # Remove sequence dimension [N, 1, D] -> [N, D]
        x = x.squeeze(1)
        
        # Generate final output logits
        out = self.lin_out(x)
        return out


# ==================== 3. Hybrid GAT-SAGE Model ====================
class HybridGATSAGE(nn.Module):
    """
    Hybrid Architecture combining GAT's attention mechanism with SAGE's efficiency
    
    Design Rationale:
    - GAT layers: Capture important drug-drug relationships through learned attention weights
    - SAGE layers: Provide computational efficiency through neighbor sampling
    - Skip connections: Enable better gradient flow and training stability
    - Alternating architecture: Balances expressiveness with computational cost
    
    This custom design leverages the strengths of both GAT (Graph Attention Networks)
    and GraphSAGE (Graph Sample and Aggregate) to create a powerful yet efficient model.
    
    Architecture Flow:
    1. Initial GAT layer with multi-head attention
    2. Alternating GAT/SAGE layers in middle
    3. Final SAGE layer for efficiency
    4. Skip connection from input to output for gradient flow
    
    Args:
        in_channels: Dimension of input node features
        hidden_channels: Dimension of hidden representations
        out_channels: Number of output classes
        num_layers: Total number of message passing layers (default: 2)
        dropout: Dropout probability for regularization (default: 0.5)
        heads: Number of attention heads for GAT layers (default: 4)
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, heads=4):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Message passing layers and batch normalization
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer: GAT with multi-head attention
        # Output dimension: hidden_channels * heads (concatenated heads)
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Middle layers: Alternate between GAT and SAGE for balanced performance
        for i in range(num_layers - 2):
            if i % 2 == 0:
                # Even layers: GAT for attention-based aggregation
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                         heads=heads, dropout=dropout))
                self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
            else:
                # Odd layers: SAGE for efficient mean aggregation
                self.convs.append(SAGEConv(hidden_channels * heads, hidden_channels * heads))
                self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Final layer: SAGE for computational efficiency
        self.convs.append(SAGEConv(hidden_channels * heads, out_channels))
        
        # Skip connection: Direct path from input to output
        # Helps with gradient flow and enables the network to learn identity mappings if needed
        self.skip = Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        """
        Forward pass with skip connections
        
        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Graph edge indices [2, num_edges]
            
        Returns:
            Output logits for classification [num_nodes, out_channels]
        """
        # Compute skip connection: direct transformation from input to output space
        x_skip = self.skip(x)
        
        # Main path: Process through all intermediate layers
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)  # Message passing
            x = self.bns[i](x)  # Batch normalization
            x = F.elu(x)  # ELU activation (smooth, non-zero gradient for negative values)
            x = F.dropout(x, p=self.dropout, training=self.training)  # Regularization
        
        # Final output layer (no activation or normalization)
        x = self.convs[-1](x, edge_index)
        
        # Add skip connection: Combine main path with direct path
        # This helps with gradient flow and model optimization
        x = x + x_skip
        
        return x


# ==================== Model Factory ====================
def create_advanced_model(model_name, hetero_data=None, in_channels=4, 
                         hidden_channels=32, out_channels=4, 
                         num_layers=2, dropout=0.5, **kwargs):
    """
    Factory function to create advanced GNN models
    
    This function provides a unified interface for creating different types of
    advanced GNN architectures with consistent hyperparameters.
    
    Supported Models:
    - 'RGCN': Relational Graph Convolutional Network for heterogeneous graphs
    - 'HGT': Heterogeneous Graph Transformer with attention mechanism
    - 'HybridGATSAGE' or 'HYBRID': Custom hybrid architecture
    
    Args:
        model_name: Name of the model to create ('RGCN', 'HGT', 'HybridGATSAGE')
        hetero_data: Heterogeneous graph data (required for RGCN and HGT)
        in_channels: Dimension of input node features (for HybridGATSAGE)
        hidden_channels: Dimension of hidden representations (default: 32)
        out_channels: Number of output classes (default: 4)
        num_layers: Number of message passing layers (default: 2)
        dropout: Dropout probability for regularization (default: 0.5)
        **kwargs: Additional model-specific arguments:
            - num_bases: Number of basis matrices for RGCN (default: 30)
            - num_heads: Number of attention heads for HGT (default: 4)
            - heads: Number of attention heads for HybridGATSAGE (default: 4)
    
    Returns:
        Initialized model instance
        
    Raises:
        ValueError: If model_name is unknown or required arguments are missing
    """
    model_name = model_name.upper()
    
    if model_name == 'RGCN':
        if hetero_data is None:
            raise ValueError("RGCN requires hetero_data argument for initialization")
        num_bases = kwargs.get('num_bases', 30)
        return HeteroRGCN(hetero_data, hidden_channels, out_channels, 
                         num_layers, dropout, num_bases)
    
    elif model_name == 'HGT':
        if hetero_data is None:
            raise ValueError("HGT requires hetero_data argument for initialization")
        num_heads = kwargs.get('num_heads', 4)
        return HGTModel(hetero_data, hidden_channels, out_channels,
                       num_heads, num_layers, dropout)
    
    elif model_name == 'HYBRIDGATSAGE' or model_name == 'HYBRID':
        heads = kwargs.get('heads', 4)
        return HybridGATSAGE(in_channels, hidden_channels, out_channels,
                            num_layers, dropout, heads)
    
    else:
        raise ValueError(f"Unknown advanced model: {model_name}. "
                        f"Supported models: RGCN, HGT, HybridGATSAGE")


# ==================== Heterogeneous Trainer ====================
class HeteroGNNTrainer:
    """
    Trainer class for heterogeneous GNN models (RGCN, HGT)
    
    This trainer handles the complete training pipeline for heterogeneous graph models,
    including training loops, validation, early stopping, and model checkpointing.
    
    Features:
    - Automatic device management (GPU/CPU)
    - Class weight support for imbalanced datasets
    - Training history tracking
    - Early stopping with patience
    - Best model checkpointing
    - Comprehensive evaluation metrics
    
    Args:
        model: The GNN model to train (RGCN or HGT)
        device: Device for training ('cuda' or 'cpu', auto-detected by default)
        class_weights: Optional class weights for handling imbalanced datasets
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights = class_weights.to(device)
        
        # Training history for tracking metrics over epochs
        self.history = {
            'train_loss': [],  # Training loss per epoch
            'train_acc': [],   # Training accuracy per epoch
            'val_loss': [],    # Validation loss per epoch
            'val_acc': [],     # Validation accuracy per epoch
            'val_f1': []       # Validation F1 score per epoch
        }
        
    def train_epoch(self, hetero_data, optimizer, mask):
        """
        Train the model for one epoch on heterogeneous graph data
        
        Args:
            hetero_data: Heterogeneous graph data object
            optimizer: PyTorch optimizer
            mask: Boolean mask for training nodes
            
        Returns:
            tuple: (loss, accuracy) for this epoch
        """
        self.model.train()  # Set model to training mode
        optimizer.zero_grad()  # Clear gradients
        
        # Forward pass through the model
        out = self.model(hetero_data.x_dict, hetero_data.edge_index_dict)
        
        # Compute cross-entropy loss with optional class weights
        loss = F.cross_entropy(out[mask], hetero_data['drug'].y[mask], 
                              weight=self.class_weights)
        
        # Backward pass and optimization
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        
        # Calculate training accuracy
        from sklearn.metrics import accuracy_score
        pred = out[mask].argmax(dim=1)  # Get predicted classes
        acc = accuracy_score(hetero_data['drug'].y[mask].cpu(), pred.cpu())
        
        return loss.item(), acc
    
    @torch.no_grad()
    def evaluate(self, hetero_data, mask):
        """
        Evaluate the model on validation/test data
        
        This method runs in inference mode (no gradient computation) for efficiency.
        
        Args:
            hetero_data: Heterogeneous graph data object
            mask: Boolean mask for evaluation nodes
            
        Returns:
            tuple: (loss, accuracy, f1_score, true_labels, predicted_labels)
        """
        from sklearn.metrics import accuracy_score, f1_score
        
        self.model.eval()  # Set model to evaluation mode
        
        # Forward pass
        out = self.model(hetero_data.x_dict, hetero_data.edge_index_dict)
        
        # Compute loss
        loss = F.cross_entropy(out[mask], hetero_data['drug'].y[mask],
                              weight=self.class_weights)
        
        # Get predictions and convert to numpy for sklearn metrics
        pred = out[mask].argmax(dim=1)
        y_true = hetero_data['drug'].y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')  # Macro-averaged F1
        
        return loss.item(), acc, f1, y_true, y_pred
    
    def fit(self, hetero_data, train_mask, val_mask, epochs=200, lr=0.01,
            weight_decay=5e-4, patience=50, verbose=True):
        """
        Train the model with early stopping and best model checkpointing
        
        Training loop with the following features:
        - Adam optimizer with weight decay (L2 regularization)
        - Early stopping based on validation accuracy
        - Best model state preservation
        - Training history tracking
        - Optional progress bar
        
        Args:
            hetero_data: Heterogeneous graph data object
            train_mask: Boolean mask for training nodes
            val_mask: Boolean mask for validation nodes
            epochs: Maximum number of training epochs (default: 200)
            lr: Learning rate (default: 0.01)
            weight_decay: L2 regularization coefficient (default: 5e-4)
            patience: Number of epochs to wait before early stopping (default: 50)
            verbose: Whether to show progress bar and messages (default: True)
            
        Returns:
            dict: Training history containing losses and metrics
        """
        from tqdm import tqdm
        
        # Move data to device
        hetero_data = hetero_data.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        
        # Initialize Adam optimizer with weight decay
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Early stopping variables
        best_val_acc = 0
        patience_counter = 0
        
        # Setup progress bar
        if verbose:
            pbar = tqdm(range(epochs), desc='Training')
        else:
            pbar = range(epochs)
        
        # Training loop
        for epoch in pbar:
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(hetero_data, optimizer, train_mask)
            
            # Evaluate on validation set
            val_loss, val_acc, val_f1, _, _ = self.evaluate(hetero_data, val_mask)
            
            # Record metrics in history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            # Check for improvement and update best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model state (on CPU to save GPU memory)
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch} - no improvement for {patience} epochs")
                break
            
            # Update progress bar with current metrics
            if verbose:
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}',
                    'val_f1': f'{val_f1:.4f}'
                })
        
        # Load best model state
        self.model.load_state_dict(self.best_model_state)
        self.model.to(self.device)
        
        return self.history
    
    def test(self, hetero_data, test_mask, label_names=None):
        """
        Test the trained model and generate comprehensive evaluation report
        
        Args:
            hetero_data: Heterogeneous graph data object
            test_mask: Boolean mask for test nodes
            label_names: Optional dictionary mapping class indices to names
            
        Returns:
            dict: Test results including:
                - test_loss: Cross-entropy loss on test set
                - test_acc: Accuracy on test set
                - test_f1: Macro F1 score on test set
                - y_true: True labels
                - y_pred: Predicted labels
                - report: Detailed classification report string
        """
        from sklearn.metrics import classification_report
        
        # Move data to device
        hetero_data = hetero_data.to(self.device)
        test_mask = test_mask.to(self.device)
        
        # Evaluate on test set
        test_loss, test_acc, test_f1, y_true, y_pred = self.evaluate(hetero_data, test_mask)
        
        # Prepare class names for classification report
        if label_names is not None:
            target_names = [label_names[i] for i in sorted(label_names.keys())]
        else:
            target_names = None
        
        # Generate detailed classification report
        report = classification_report(y_true, y_pred, target_names=target_names)
        
        # Package results
        results = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'report': report
        }
        
        return results


# ==================== Module Test ====================
if __name__ == "__main__":
    print("Advanced GNN Models for Extra Credit")
    print("=" * 50)
    print("Available models:")
    print("  1. RGCN - Relational Graph Convolutional Network")
    print("  2. HGT - Heterogeneous Graph Transformer")
    print("  3. HybridGATSAGE - Hybrid GAT-SAGE architecture")
    print("=" * 50)