import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data
import warnings
import logging 

import time

import os

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class GRBSANeuralPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, dropout, num_classes, num_node_types, num_edge_types, processing_steps):
        super().__init__()
        
        # Original graph processing components
        self.graph_processor = MolecularGraphEncoder(
            in_dim, hidden_dim, num_layers, num_heads, 
            dropout, num_classes, num_node_types, 
            num_edge_types, processing_steps
        )
        
        # Domain adaptation components
        self.domain_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim//2, 1)
            ) for _ in range(4)  # For main + 3 test datasets
        ])
        
        # Original classifier
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data, domain_idx=0, alpha=1.0):
        # Process graph features
        graph_features = self.graph_processor(data)
        
        # Domain classification with gradient reversal
        domain_features = GradientReversalLayer.apply(graph_features, alpha)
        domain_output = self.domain_classifiers[domain_idx](domain_features)
        
        # Class prediction - ensure this returns a tensor, not a tuple
        class_output = self.graph_classifier(graph_features).squeeze(-1)
        
        # Make sure we're not returning nested tuples
        if isinstance(class_output, tuple):
            class_output = class_output[0]
            
        return class_output, domain_output
    

class MolecularGraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, num_heads, dropout, num_classes, num_node_types, num_edge_types, processing_steps):
        super().__init__()
        
        # Node feature encoding
        self.node_embedding = nn.Linear(in_dim, hidden_dim)
        
        # Graph Attention layers
        self.gat_layers = nn.ModuleList([
            GATConv(
                hidden_dim, 
                hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Node type embeddings
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_dim)
        
        # Global attention pooling
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Initial node feature embedding
        h = self.node_embedding(x)
        h = self.dropout(h)
        
        # Add node type information if available
        if hasattr(data, 'node_types'):
            node_type_embed = self.node_type_embedding(data.node_types)
            h = h + node_type_embed
        
        # Process through GAT layers
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            h_prev = h
            h = gat(h, edge_index)
            h = norm(h)
            h = h + h_prev  # Residual connection
            h = F.relu(h)
            h = self.dropout(h)
        
        # Global pooling
        attention_weights = self.global_attention(h)
        attention_weights = F.softmax(attention_weights, dim=0)
        h_graph = torch.sum(h * attention_weights, dim=0)
        
        # Additional global mean pooling
        h_graph = h_graph + global_mean_pool(h, batch)
        
        return h_graph
    

    
class DomainAdaptiveLayer(nn.Module):
    def __init__(self, feature_dim, domains=4):
        super().__init__()
        self.domain_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim//2),
                nn.ReLU(),
                nn.Linear(feature_dim//2, 1)
            ) for _ in range(domains)
        ])
        self.domain_adapters = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(domains)
        ])
        
    def forward(self, x, domain_idx=0, alpha=1.0):
        # Domain classification with gradient reversal
        domain_pred = self.domain_classifiers[domain_idx](GradientReversalLayer.apply(x, alpha))
        
        # Feature adaptation
        adapted_features = self.domain_adapters[domain_idx](x)
        return adapted_features, domain_pred
    

