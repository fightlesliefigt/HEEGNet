# HEEGNET: HYPERBOLIC EMBEDDINGS FOR EEG

This repository contains code accompanying the ICLR 2026 paper *[HEEGNET: HYPERBOLIC EMBEDDINGS FOR EEG](https://openreview.net/forum?id=CNDNRjpVIL)*.

## File list

The following files are provided in this repository:

`demo.ipynb` A jupyter notebook to train and evaluate the proposed model.

`nets` A folder containing the HEEGNet hybrid hyperbolic deep learning framework.

`Geometry`, `hsssw`, `lib`  three folders contain hyperbolic space operations. 'Geometry' defines hyperbolic neural networks, 'hsssw' defines optimal transport on the hyperbolic manifold, and `lib` defines the Lorentz batch normalization operation.

`pretrained_models` A folder containing the pretrained model for the source domains of the Faced dataset, enabling immediate source-free unsupervised domain adaptation to validate reported results.


## Requirements

All dependencies are managed with the `conda` package manager.
Please follow the user guide to [install](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) `conda`.

Once the setup is completed, the dependencies can be installed in a new virtual environment.

Open a jupyter notebook and run it.


## Emotion recognition Experiment

In order to use this dataset, the download folder `Processed_data` (download from this URL: [https://www.synapse.org/#!Synapse:syn50615881]) 
is required, containing the following files:
Processed_data/
├── sub000.pkl
├── sub001.pkl
├── sub002.pkl
└── ...


More detail instructions  are described in the `demo.ipynb` notebook.




