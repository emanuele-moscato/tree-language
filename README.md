# Tree-based language generation

**Idea:** investigate a language (sequences of symbols) based on a tree-tructured generative model and perform inference on the generated sequences.

# Reproducibility

Main scripts used for model training (MLM and root inference):
- `scripts/Transformer_MLM.py`
- `scripts/Transformer_MLM_factorized.py`
- `scripts/Transformer_wPE.py`
- `scripts/Transformer_wPE_factorized.py`

Implementation of the BP algorithm can be found in the module:
- `modules/BeliefPropagation.py`
