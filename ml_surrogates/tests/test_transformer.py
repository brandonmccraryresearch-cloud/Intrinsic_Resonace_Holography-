"""
Tests for Phase 3 Transformer Components

Tests for:
- attention_modules.py
- holographic_encoder.py
- resonance_decoder.py
- irh_transformer.py
"""

import pytest
import numpy as np

# Import components
try:
    from ml_surrogates.models import (
        IRHTransformer,
        HolographicEncoder,
        ResonanceDecoder,
        MultiHeadAttention,
        GraphAttention,
        PositionalEncoding
    )
    from ml_surrogates.engines import CouplingState, HolographicState
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from models import (
        IRHTransformer,
        HolographicEncoder,
        ResonanceDecoder,
        MultiHeadAttention,
        GraphAttention,
        PositionalEncoding
    )
    from engines import CouplingState, HolographicState


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""
    
    def test_initialization(self):
        """Test basic initialization."""
        mha = MultiHeadAttention(embed_dim=128, num_heads=8)
        assert mha.embed_dim == 128
        assert mha.num_heads == 8
        assert mha.head_dim == 16
    
    def test_forward_pass(self):
        """Test forward pass."""
        mha = MultiHeadAttention(embed_dim=128, num_heads=8)
        query = np.random.randn(2, 10, 128)
        output = mha(query)
        assert output.shape == (2, 10, 128)
    
    def test_cross_attention(self):
        """Test cross-attention with different K, V."""
        mha = MultiHeadAttention(embed_dim=128, num_heads=8)
        query = np.random.randn(2, 5, 128)
        key_value = np.random.randn(2, 10, 128)
        output = mha(query, key=key_value, value=key_value)
        assert output.shape == (2, 5, 128)


class TestGraphAttention:
    """Tests for GraphAttention."""
    
    def test_initialization(self):
        """Test basic initialization."""
        gat = GraphAttention(node_dim=64, edge_dim=32, num_heads=4)
        assert gat.node_dim == 64
        assert gat.edge_dim == 32
        assert gat.num_heads == 4
    
    def test_forward_pass(self):
        """Test forward pass."""
        gat = GraphAttention(node_dim=64, edge_dim=32, num_heads=4)
        
        num_nodes, num_edges = 10, 15
        node_features = np.random.randn(num_nodes, 64)
        edge_features = np.random.randn(num_edges, 32)
        adjacency = np.random.randint(0, num_nodes, size=(num_edges, 2))
        
        output = gat(node_features, edge_features, adjacency)
        assert output.shape == (num_nodes, 64)


class TestHolographicEncoder:
    """Tests for HolographicEncoder."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = HolographicEncoder(embed_dim=128, num_layers=3, num_heads=4)
        assert encoder.embed_dim == 128
        assert encoder.num_layers == 3
        assert encoder.num_heads == 4
    
    def test_encode_state(self):
        """Test encoding holographic state."""
        encoder = HolographicEncoder(embed_dim=128)
        
        # Create test state
        initial = CouplingState(10.0, 10.0, 10.0, 1.0)
        state = HolographicState(initial)
        
        for i in range(5):
            new_state = CouplingState(10.0 - i, 10.0 - i, 10.0 - i, 1.0 - 0.1*i)
            state.add_rg_step(new_state, beta_functions=(-0.5, -0.5, -0.5))
        
        encoding, metadata = encoder(state)
        
        assert encoding.shape[0] == 6  # Initial + 5 steps
        assert encoding.shape[1] == 128
        assert metadata['num_nodes'] == 6
    
    def test_batch_encoding(self):
        """Test batch encoding."""
        encoder = HolographicEncoder(embed_dim=128)
        
        states = [
            HolographicState(CouplingState(10.0, 10.0, 10.0, 1.0)),
            HolographicState(CouplingState(20.0, 20.0, 20.0, 1.0))
        ]
        
        encodings, metadatas = encoder.encode_batch(states)
        
        assert len(encodings) == 2
        assert len(metadatas) == 2


class TestResonanceDecoder:
    """Tests for ResonanceDecoder."""
    
    def test_initialization(self):
        """Test decoder initialization."""
        decoder = ResonanceDecoder(embed_dim=128, num_layers=3, num_heads=8)
        assert decoder.embed_dim == 128
        assert decoder.num_layers == 3
        assert decoder.num_heads == 8
    
    def test_forward_pass(self):
        """Test decoder forward pass."""
        decoder = ResonanceDecoder(embed_dim=128)
        
        # Dummy encoded state
        encoded_state = np.random.randn(10, 128)
        
        predictions = decoder(encoded_state)
        
        assert 'couplings' in predictions
        assert 'is_fixed_point' in predictions
        assert 'action' in predictions
        assert predictions['couplings'].shape == (3,)
        assert 0 <= predictions['is_fixed_point'] <= 1
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        decoder = ResonanceDecoder(embed_dim=128)
        
        # Batch of encoded states
        batch_size = 4
        encoded_batch = np.random.randn(batch_size, 10, 128)
        
        predictions = decoder(encoded_batch)
        
        assert predictions['couplings'].shape == (batch_size, 3)
        assert predictions['is_fixed_point'].shape == (batch_size, 1)


class TestIRHTransformer:
    """Tests for complete IRHTransformer model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = IRHTransformer(
            embed_dim=128,
            encoder_layers=3,
            decoder_layers=3
        )
        
        assert model.embed_dim == 128
        assert not model.trained
    
    def test_forward_pass(self):
        """Test model forward pass."""
        model = IRHTransformer(embed_dim=128)
        
        # Create test state
        initial = CouplingState(10.0, 10.0, 10.0, 1.0)
        state = HolographicState(initial)
        
        for i in range(3):
            new_state = CouplingState(10.0 - i, 10.0 - i, 10.0 - i, 1.0 - 0.1*i)
            state.add_rg_step(new_state)
        
        predictions = model(state)
        
        assert 'couplings' in predictions
        assert 'is_fixed_point' in predictions
        assert 'action' in predictions
    
    def test_predict_final_state(self):
        """Test final state prediction."""
        model = IRHTransformer(embed_dim=128)
        
        initial = CouplingState(10.0, 10.0, 10.0, 1.0)
        
        final_state = model.predict_final_state(initial, target_scale=0.1)
        
        assert isinstance(final_state, CouplingState)
        assert final_state.k == 0.1
    
    def test_predict_fixed_point(self):
        """Test fixed point prediction."""
        model = IRHTransformer(embed_dim=128)
        
        initial = CouplingState(10.0, 10.0, 10.0, 1.0)
        state = HolographicState(initial)
        
        fp, confidence = model.predict_fixed_point(state)
        
        assert 0 <= confidence <= 1
        if fp is not None:
            assert isinstance(fp, CouplingState)
    
    def test_batch_prediction(self):
        """Test batch prediction."""
        model = IRHTransformer(embed_dim=128)
        
        states = [
            HolographicState(CouplingState(10.0, 10.0, 10.0, 1.0)),
            HolographicState(CouplingState(20.0, 20.0, 20.0, 1.0))
        ]
        
        predictions = model.predict_batch(states)
        
        assert len(predictions) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
