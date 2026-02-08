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

### Cross-Subject Cross-Validation Protocol

For evaluation, we adopt a **group-wise cross-validation strategy** where:
- the grouping variable is `subject_id`,
- all sessions of the same subject are strictly assigned to either the training set or the test set,
- no subject appears in both splits.

This protocol corresponds to a **cross-subject generalization setting**, while still allowing **session-level domain alignment** within the training data.

### Cross-Session Cross-Validation Protocol 

In addition to the cross-subject setting, we also support a **cross-session generalization setting**.  
In this case:
- the grouping variable is `session_id`,
- all samples from one session are used as the test domain,
- remaining sessions are used for training.

This protocol evaluates the model’s ability to generalize across different recording sessions **within the same subject**.


### Design Rationale in cross-subject setting
Although one could alternatively define each subject as a domain, we note that using **session-wise domains** leads to better performance.  
This is because session-wise domain modeling explicitly addresses **cross-session non-stationarity**, which is also a source of distribution shift in EEG data.

### Example of the usage
```
#network and training configuration
cfg = dict(
    epochs = 100,
    batch_size_train = 50,
    domains_per_batch = 5,
    validation_size = 0.2,
    evaluation = 'inter-subject'
    dtype = torch.float64,
    training=True, 
    lr=0.001,
    input_align= True, # To perform the input space alignment or not
    weight_decay=1e-4,
    swd_weight=0.01, # Loss weight for the stage two gaussian alignment
    mdl_kwargs = dict( 
    bnorm_dispersion=bn.BatchNormDispersion.SCALAR, 
    domain_adaptation=True # To perform domains-specific batch normalization on hyperbolic maniold or not
)
)

# add datadependen model kwargs
mdl_kwargs = deepcopy(cfg['mdl_kwargs'])
mdl_kwargs['num_classes'] = n_classes
mdl_kwargs['num_electrodes'] = X.shape[1]
mdl_kwargs['chunk_size'] = X.shape[2]
mdl_kwargs['domains'] = domain.unique()

# create the model and training configuration
model = HEEGNet(device=device,dtype=cfg['dtype'],**mdl_kwargs).to(device=device, dtype=cfg['dtype'])

# early stopping
es = EarlyStopping(metric='val_loss', higher_is_better=False, patience=20, verbose=False)

#  hyperbolic maniold batch normalization scheduler
bn_sched = MomentumBatchNormScheduler(
    epochs=cfg['epochs']-10,
    bs0=cfg['batch_size_train'],
    bs=cfg['batch_size_train']/cfg['domains_per_batch'], 
    tau0=0.85
)

# create the trainer
trainer = Trainer(
    max_epochs= cfg['epochs'],
    min_epochs= 70,
    callbacks=[es,bn_sched],
    loss=torch.nn.CrossEntropyLoss(),
    device=device, 
    dtype=torch.float64,
    swd_weight=cfg['swd_weight'],
    lr=cfg['lr'],
)

```



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
```
@inproceedings{li2026heegnet,
  title={HEEGNet: Hyperbolic Embeddings for EEG},
  author={Li, Shanglin and Chu, Shiwen and Ko{\c{c}}, Okan and Ding, Yi and Zhao, Qibin and Kawanabe, Motoaki and Chen, Ziheng},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```




