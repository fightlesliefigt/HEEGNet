# HEEGNET: HYPERBOLIC EMBEDDINGS FOR EEG

This repository contains code accompanying the ICLR 2026 paper *[HEEGNET: HYPERBOLIC EMBEDDINGS FOR EEG](https://openreview.net/forum?id=CNDNRjpVIL)*.

## File list

The following files are provided in this repository:

`demo.ipynb` A jupyter notebook to train and evaluate the proposed model.

`nets` A folder containing the HEEGNet hybrid hyperbolic deep learning framework.

`Geometry`, `hsssw`, `lib`  three folders contain hyperbolic space operations. `lib` defines hyperbolic neural networks, 'hsssw' defines optimal transport on the hyperbolic manifold, and 'Geometry' defines the Lorentz batch normalization operation.

`pretrained_models` A folder containing the pretrained model for the source domains of the Faced dataset, enabling immediate source-free unsupervised domain adaptation to validate reported results.

More detail instructions  are described in the `demo.ipynb` notebook.

## Prepare the python virtual environment

All dependencies are managed with the `conda` package manager.
Please follow the user guide to [install](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) `conda`.

Once the setup is completed, the dependencies can be installed in a new virtual environment.

Open a jupyter notebook and run it.



## Apply HEEGNet to other datasets
### Domain-Specific Normalization
In this project, we define a **domain** as a **subject–session pair**, i.e., each recording session of each subject is treated as an independent domain.

Formally, each EEG sample is associated with:
- `subject_id`: the biological participant,
- `session_id`: the recording session index of that subject,
- `domain_id = (subject_id, session_id)`.

During training, **Domain-Specific Batch Normalization (DSBN / DSMDBN)** maintains separate running statistics (mean and variance) for each `domain_id`.  
This allows the model to explicitly capture distribution shifts across different recording sessions.

### Cross-Validation Protocol

For evaluation, we adopt a **group-wise cross-validation strategy** where:
- the grouping variable is `subject_id`,
- all sessions of the same subject are strictly assigned to either the training set or the test set,
- no subject appears in both splits.

This protocol corresponds to a **cross-subject generalization setting**, while still allowing **session-level domain alignment** within the training data.

### Design Rationale
Although one could alternatively define each subject as a domain, we empirically find that using **session-wise domains** leads to better performance.  
This is because session-wise domain modeling explicitly addresses **cross-session non-stationarity**, which is also a source of distribution shift in EEG data.


## Emotion recognition Experiment
In order to use this dataset, the download folder `Processed_data` (download from this URL: [https://www.synapse.org/#!Synapse:syn50615881]) 
is required, containing the following files:
Processed_data/
├── sub000.pkl
├── sub001.pkl
├── sub002.pkl
└── ...

## CBCR License
| Permissions | Limitations | Conditions |
| :---         |     :---:      |          :---: |
| :white_check_mark: Modification   | :x: Commercial use   | :warning: License and copyright notice   |
| :white_check_mark: Distribution     |       |      |
| :white_check_mark: Private use     |        |      |


## Cite
Please cite our paper if you use our code in your own work:
@inproceedings{li2026heegnet,
  title={HEEGNet: Hyperbolic Embeddings for EEG},
  author={Li, Shanglin and Chu, Shiwen and Ko{\c{c}}, Okan and Ding, Yi and Zhao, Qibin and Kawanabe, Motoaki and Chen, Ziheng},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}





