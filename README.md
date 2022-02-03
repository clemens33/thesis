# Master Thesis in AI 

If you have any questions regarding my results or work I'm happy to answer them! Just contact me over GitHub or open an issue.

## Thesis

For my written master thesis including detailed results, experiment descriptions and theoretical background please refer to my work "[Attentive Tabular Learning in Context of Drug Discovery](https://github.com/clemens33/thesis-ai/blob/main/Kriechbaumer_Master_Thesis_AI_final.pdf)".

## Project

This project contains all the source for my master thesis including:   

- [tabnet](./src/tabnet): a pure pytorch based tabnet reimplementation including various attentive types including sparsemax, entmax-15, alpha-entmax and others. 
- [tabnet_lightning](./src/tabnet_lightning): a pytorch lightning wrapper for various tasks, at the moment mostly focusing on classification tasks
- [tests](./tests): pytest based test suite for all relevant implementations. The TabNet reimplementation is quite rigorously tested. 
- [baseline](.src/baseline): reference and baseline model code including a MLP, RF and GBDT for comparison used during experiments.
- [datasets](.src/datasets): pytorch lighting based data modules for various datasets mostly within the drug discovery field. Focuses on local caching and uses multi-processing wherever possible. Includes various molecule [featurizer](.src/datasets/featurizer.py) implementations which also can calculate the atomic contribution of feature dimension (basically the reverse path feature to atom in molecule). 

## Results

Access to all experiments is provided at [mlflow.kriechbaumer.at](https://mlflow.kriechbaumer.at) (might be taken down in the future).

