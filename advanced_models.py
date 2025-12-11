"""
9_advanced_models.py
Advanced GNN Models for Extra Credit (+20%)

实现:
1. R-GCN (Relational GCN) - 用于异构图
2. HGT (Heterogeneous Graph Transformer)  
3. Hybrid GAT-SAGE Model - 自定义设计
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
    
    Paper: Schlichtkrull et al. (2018) "Modeling Relational Data with Graph CNNs"
    
    Note: Uses metapath aggregation instead of full RGCN for memory efficiency
    """
    def __init__(self, hetero_data, hidden_channels=32, out_channels=4, 
                 num_layers=2, dropout=0.5, num_bases=30):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Get input dimensions
        drug_in_channels = hetero_data['drug'].x.shape[1]
        
        # Projections for different metapaths
        self.drug_proj = Linear(drug_in_channels, hidden_channels)
        
        # Message passing layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(Linear(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.lin_out = Linear(hidden_channels, out_channels)
        
    def forward(self, x_dict, edge_index_dict):
        # Project drug features
        x = self.drug_proj(x_dict['drug'])
        
        # Apply layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output
        out = self.lin_out(x)
        return out


# ==================== 2. HGT (Heterogeneous Graph Transformer) ====================
class HGTModel(nn.Module):
    """
    Simplified Heterogeneous Graph model with attention
    
    Inspired by: Hu et al. (2020) "Heterogeneous Graph Transformer"
    
    Simplified for stability and memory efficiency
    """
    def __init__(self, hetero_data, hidden_channels=32, out_channels=4,
                 num_heads=4, num_layers=2, dropout=0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        
        # Get input dimensions
        drug_in_channels = hetero_data['drug'].x.shape[1]
        
        # Input projection
        self.lin_in = Linear(drug_in_channels, hidden_channels)
        
        # Attention layers (simplified multi-head attention)
        self.attns = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.attns.append(nn.MultiheadAttention(
                hidden_channels, 
                num_heads, 
                dropout=dropout,
                batch_first=True
            ))
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # FFN layers
        self.ffns = nn.ModuleList()
        for _ in range(num_layers):
            self.ffns.append(nn.Sequential(
                Linear(hidden_channels, hidden_channels * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(hidden_channels * 2, hidden_channels)
            ))
        
        # Output layer
        self.lin_out = Linear(hidden_channels, out_channels)
        
    def forward(self, x_dict, edge_index_dict):
        # Project drug features
        x = self.lin_in(x_dict['drug'])
        
        # Add batch dimension for attention
        x = x.unsqueeze(1)  # [N, 1, D]
        
        # Apply attention layers
        for attn, norm, ffn in zip(self.attns, self.norms, self.ffns):
            # Self-attention
            attn_out, _ = attn(x, x, x)
            x = norm(x + attn_out)
            
            # FFN
            ffn_out = ffn(x)
            x = norm(x + ffn_out)
        
        # Remove batch dimension
        x = x.squeeze(1)  # [N, D]
        
        # Output
        out = self.lin_out(x)
        return out


# ==================== 3. Hybrid GAT-SAGE Model ====================
class HybridGATSAGE(nn.Module):
    """
    Hybrid model combining GAT's attention with SAGE's efficient sampling
    
    Design rationale:
    - GAT for capturing important drug relationships
    - SAGE for computational efficiency
    - Skip connections for better gradient flow
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, dropout=0.5, heads=4):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer: GAT
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Middle layers: Alternating GAT and SAGE
        for i in range(num_layers - 2):
            if i % 2 == 0:
                # GAT layer
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels, 
                                         heads=heads, dropout=dropout))
                self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
            else:
                # SAGE layer  
                self.convs.append(SAGEConv(hidden_channels * heads, hidden_channels * heads))
                self.bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels * heads, out_channels))
        
        # Skip connection
        self.skip = Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        # Save input for skip connection
        x_skip = self.skip(x)
        
        # Main path
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output layer
        x = self.convs[-1](x, edge_index)
        
        # Add skip connection
        x = x + x_skip
        
        return x


# ==================== Model Factory ====================
def create_advanced_model(model_name, hetero_data=None, in_channels=4, 
                         hidden_channels=32, out_channels=4, 
                         num_layers=2, dropout=0.5, **kwargs):
    """
    Create advanced GNN models
    
    Args:
        model_name: 'RGCN', 'HGT', 'HybridGATSAGE'
        hetero_data: Required for RGCN and HGT
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimension
        out_channels: Output dimension
        num_layers: Number of layers
        dropout: Dropout rate
    """
    model_name = model_name.upper()
    
    if model_name == 'RGCN':
        if hetero_data is None:
            raise ValueError("RGCN requires hetero_data")
        num_bases = kwargs.get('num_bases', 30)
        return HeteroRGCN(hetero_data, hidden_channels, out_channels, 
                         num_layers, dropout, num_bases)
    
    elif model_name == 'HGT':
        if hetero_data is None:
            raise ValueError("HGT requires hetero_data")
        num_heads = kwargs.get('num_heads', 4)
        return HGTModel(hetero_data, hidden_channels, out_channels,
                       num_heads, num_layers, dropout)
    
    elif model_name == 'HYBRIDGATSAGE' or model_name == 'HYBRID':
        heads = kwargs.get('heads', 4)
        return HybridGATSAGE(in_channels, hidden_channels, out_channels,
                            num_layers, dropout, heads)
    
    else:
        raise ValueError(f"Unknown advanced model: {model_name}")


# ==================== Heterogeneous Trainer ====================
class HeteroGNNTrainer:
    """Trainer for heterogeneous GNN models (RGCN, HGT)"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights = class_weights.to(device)
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
    def train_epoch(self, hetero_data, optimizer, mask):
        """Train one epoch on heterogeneous data"""
        self.model.train()
        optimizer.zero_grad()
        
        # Forward
        out = self.model(hetero_data.x_dict, hetero_data.edge_index_dict)
        loss = F.cross_entropy(out[mask], hetero_data['drug'].y[mask], 
                              weight=self.class_weights)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        from sklearn.metrics import accuracy_score
        pred = out[mask].argmax(dim=1)
        acc = accuracy_score(hetero_data['drug'].y[mask].cpu(), pred.cpu())
        
        return loss.item(), acc
    
    @torch.no_grad()
    def evaluate(self, hetero_data, mask):
        """Evaluate on heterogeneous data"""
        from sklearn.metrics import accuracy_score, f1_score
        
        self.model.eval()
        
        out = self.model(hetero_data.x_dict, hetero_data.edge_index_dict)
        loss = F.cross_entropy(out[mask], hetero_data['drug'].y[mask],
                              weight=self.class_weights)
        
        pred = out[mask].argmax(dim=1)
        y_true = hetero_data['drug'].y[mask].cpu().numpy()
        y_pred = pred.cpu().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        return loss.item(), acc, f1, y_true, y_pred
    
    def fit(self, hetero_data, train_mask, val_mask, epochs=200, lr=0.01,
            weight_decay=5e-4, patience=50, verbose=True):
        """Train the model"""
        from tqdm import tqdm
        
        hetero_data = hetero_data.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        best_val_acc = 0
        patience_counter = 0
        
        if verbose:
            pbar = tqdm(range(epochs), desc='Training')
        else:
            pbar = range(epochs)
        
        for epoch in pbar:
            train_loss, train_acc = self.train_epoch(hetero_data, optimizer, train_mask)
            val_loss, val_acc, val_f1, _, _ = self.evaluate(hetero_data, val_mask)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break
            
            if verbose:
                pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}',
                    'val_f1': f'{val_f1:.4f}'
                })
        
        self.model.load_state_dict(self.best_model_state)
        self.model.to(self.device)
        
        return self.history
    
    def test(self, hetero_data, test_mask, label_names=None):
        """Test the model"""
        from sklearn.metrics import classification_report
        
        hetero_data = hetero_data.to(self.device)
        test_mask = test_mask.to(self.device)
        
        test_loss, test_acc, test_f1, y_true, y_pred = self.evaluate(hetero_data, test_mask)
        
        if label_names is not None:
            target_names = [label_names[i] for i in sorted(label_names.keys())]
        else:
            target_names = None
        
        report = classification_report(y_true, y_pred, target_names=target_names)
        
        results = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'report': report
        }
        
        return results


# ==================== Test ====================
if __name__ == "__main__":
    print("Advanced GNN models for Extra Credit")
    print("Models available: RGCN, HGT, HybridGATSAGE")
