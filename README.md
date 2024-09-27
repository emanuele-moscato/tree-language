# Code for the paper "How transformers learn structured data: insights from hierarchical filtering."

## Tree-based language generation
The script generating data following the filtered hierarchical model introduced in the paper is
- `scripts/gen_filtered_hierarchical_data.py`.

It assumes the existence of a subfolder `./data/`in which the generated data will be stored as a Numpy `.npy` container, containing the generation parameters, labels (root symbol) and sequences for a chosen number of samples. All other scripts will assume the data has been generated in this way.

## Training of transformers

We also provide modules and training scripts to train the transformer architecture studied in the paper on our data model. In any of these two tasks, the scripts output results (i.e. the training history) in a Numpy `.npy` container in a `./results/` subfolder, as well as the full PyTorch model (possibly at several checkpoints) in an appropriately named subfolder itself in `./models/`subfolder. These models can then notably be used to visualize attention maps or perform fine-tuning.

### Root inference
There are two scripts: 
- `scripts/Transformer_wPE.py` to train the model on the fully hierarchical data and measure the validation accuracy on all levels of filtering,
- `scripts/Transformer_wPE_factorized.py` to train the model on filtered data and measure the validation on the full hierarchy.

### Masked language modeling
Similarly, there are also two scripts:
- `scripts/Transformer_MLM.py` to train the model on the fully hierarchical data and measure the validation accuracy on all levels of filtering,
- `scripts/Transformer_MLM_factorized.py` to train the model on filtered data and measure the validation on the full hierarchy.

## Belief Propagation
An efficient implementation of the BP algorithm for both root inference and mask language modeling can be found in:
- `modules/BeliefPropagation.py`.
