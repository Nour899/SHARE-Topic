# SHARE-Topic
SHARE-Topic is a Bayesian model used to infer latent representations of cells, chromatin regions, and genes. SHARE-Topic is implemented using a Gibbs sampler.

The code is written specifically to be run on GPU to decrease the running time of an MCMC chain for a given number of topics.

The Gibbs sampler code is provided in a jupyter notebook.

An example on of the analysis carried from the SHARE-Topic output is done on B-lymphoma 10x Multiome dataset.

To run SHARE_topic in development version, use pip install -e . to load SHARE_topic as a module

