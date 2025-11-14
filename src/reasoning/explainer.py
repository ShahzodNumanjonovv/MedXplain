# src/reasoning/explainer.py
"""
Explanation Generator (Week 3)

Produces human-readable explanations by combining:
  - detected visual concepts
  - symbolic rules that fired
  - ancestors/parents from the concept graph

This will later be integrated with a report generator.
"""

class ExplanationGenerator:
    def __init__(self, concept_graph):
        self.graph = concept_graph

    def make_explanation(self, detected, inferred, activated_rules):
        text = []
        text.append("=== Neuro-Symbolic Explanation ===\n")

        text.append("Detected Concepts:")
        for c in sorted(detected):
            text.append(f" • {c}")

        text.append("\nInferred Concepts:")
        for c in sorted(inferred):
            parents = self.graph.get_ancestors(c)
            if parents:
                text.append(f" • {c}  (supported by: {', '.join(parents)})")
            else:
                text.append(f" • {c}")

        text.append("\nActivated Rules:")
        for rule in activated_rules:
            text.append(f" • IF ({', '.join(rule.premise)}) → {rule.conclusion}")

        return "\n".join(text)
