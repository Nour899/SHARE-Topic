# SHARE-Topic


SHARE-Topic is a Bayesian model used to infer latent representations of cells, chromatin regions, and genes. SHARE-Topic is implemented using a Gibbs sampler.

![Model](https://github.com/Nour899/SHARE-Topic/blob/main/figures/SHARE-topic_workflow.wbep?raw=true)


The code is written specifically to be run on GPU to decrease the running time of an MCMC chain for a given number of topics.

An example on of the analysis carried from the SHARE-Topic output is done on B-lymphoma 10x Multiome dataset.

Download the processed data from https://zenodo.org/records/10418760 to the "data" folder. 

To run SHARE_topic in development version, use `pip install -e .` to load SHARE_topic as a module

