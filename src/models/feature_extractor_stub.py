# src/models/feature_extractor_stub.py
"""
Feature Extractor Stub

In Week 3 we do NOT train BiomedCLIP,
but we need a placeholder for feature extraction.

Later we will replace this file with real BiomedCLIP inference.
"""

import torch

class FeatureExtractorStub:
    def __init__(self, feature_dim=768):
        self.feature_dim = feature_dim

    def extract(self, image_tensor):
        """
        Returns random vectors to simulate backbone output.
        This allows us to test the concept trainer end-to-end.
        """
        batch = image_tensor.shape[0]
        return torch.randn(batch, self.feature_dim)
