# src/demo/run_pipeline.py
"""
Neuro-Symbolic Pipeline Demo (Week 3)

This file shows:
  1) feature extraction (stub)
  2) concept prediction
  3) symbolic rule inference
  4) explanation generation
"""

import torch
from models.feature_extractor_stub import FeatureExtractorStub
from training.concept_trainer import ConceptHead
from reasoning.concept_graph import ConceptGraph
from reasoning.symbolic_rules import Rule, RuleEngine
from reasoning.explainer import ExplanationGenerator

def build_graph_and_rules():
    graph = ConceptGraph()

    # Minimal radiology concept ontology
    graph.add_concept("Opacity", "finding")
    graph.add_concept("Consolidation", "condition")
    graph.add_concept("Pneumonia", "diagnosis")

    graph.add_relation("Opacity", "Consolidation")
    graph.add_relation("Consolidation", "Pneumonia")

    rules = [
        Rule(["Opacity"], "Consolidation"),
        Rule(["Consolidation"], "Pneumonia")
    ]

    rule_engine = RuleEngine(rules)
    return graph, rule_engine, rules

def run_demo():
    feature_extractor = FeatureExtractorStub()

    # Example "image input"
    fake_image = torch.randn(1, 3, 224, 224)
    visual_feats = feature_extractor.extract(fake_image)

    # Concept Predictor
    concept_model = ConceptHead(feature_dim=768, hidden_dim=256, num_concepts=3)
    probs = concept_model(visual_feats)[0]

    # Simulate prediction: detect concept 0 if prob > 0.5
    detected = {"Opacity"} if probs[0] > 0.5 else set()

    graph, rule_engine, rules = build_graph_and_rules()

    inferred = rule_engine.infer(detected)
    explainer = ExplanationGenerator(graph)

    explanation = explainer.make_explanation(
        detected=detected,
        inferred=inferred,
        activated_rules=[r for r in rules if r.applies(detected)]
    )

    print(explanation)

if __name__ == "__main__":
    run_demo()
