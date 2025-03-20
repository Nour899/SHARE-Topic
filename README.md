# SHARE-Topic


SHARE-Topic is a Bayesian Hierarchical model and used to infer gene-region interactions from single-cell multi-omics data. SHARE-Topic provides latent representations for cells, genes, and accessible chromatin regions.  

![Model](https://github.com/Nour899/SHARE-Topic/blob/main/figures/SHARE-topic_workflow.webp)


SHARE-Topic is implemented using a Gibbs Sampler. The code is written specifically to run on GPU to decrease the running time of an MCMC chain for a given number of topics.

An example on of the analysis carried from the SHARE-Topic output is done on B-lymphoma 10x Multiome dataset.

Download the processed data from https://zenodo.org/records/10418760 to the "data" folder. 

To run SHARE_topic in development version, use `pip install -e .` to load SHARE_topic as a module

