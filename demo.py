# %%
import torch
import sklearn
import pandas as pd
from nets.utils.data import StratifiedDomainDataLoader, DomainDataset
from nets.trainer import Trainer
from nets.callbacks import EarlyStopping, MomentumBatchNormScheduler
from nets.model import HEEGNet
import nets.functionals as fn
from copy import deepcopy
import nets.batchnorm as bn
import numpy as np

# %% [markdown]
# ## configuration for alignment
# 
# input_align, set 'True' to perform euclidian alignment in the input space
# 
# domain_adaptation, set 'True' to perform the moments alignment
# 
# swd_weight control the hhsw loss value, set zero to off the feature distribution alignment

# %%
# network and training configuration
cfg = dict(
    epochs = 100,
    batch_size_train = 50,
    domains_per_batch = 5,
    validation_size = 0.2,
    evaluation = 'inter-subject', 
    dtype = torch.float64,
    training=True, 
    lr=0.001,
    input_align= True, #! To perform the input space alignment or not
    weight_decay=1e-4,
    swd_weight=0.01, #! Loss weight for the stage two gaussian alignment
    mdl_kwargs = dict( 
    bnorm_dispersion=bn.BatchNormDispersion.SCALAR,
    domain_adaptation=True #! To perform domains-specific batch normalization on hyperbolic maniold or not
)
)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('GPU')
else:
    device = torch.device('cpu')
    print('CPU')

# %% [markdown]
# ## Loading data
# 
# input the data path 

# %%
# input the data path and load the data
dataset='faced'
n_classes = 9
loading_path=''
data, labels, subjects, sessions = fn.load_dataset(loading_path)

# grouping each subject-session as a domain
domain_labels = [f"{sub}_{sess}" for sub, sess in zip(subjects, sessions)]
domain = sklearn.preprocessing.LabelEncoder().fit_transform(domain_labels)
assert data.shape[0]==labels.shape[0]==subjects.shape[0]==sessions.shape[0]==domain.shape[0]
domain = torch.from_numpy(domain)
X=data.copy()

# %% [markdown]
# ## Define SFUDA function

# %%
def sfuda_offline(dataset : DomainDataset, model : HEEGNet):
    model.eval()
    model.domainadapt_finetune(dataset.features.to(dtype=cfg['dtype'], device=device), dataset.labels.to(device=device), dataset.domains, None)

# %% [markdown]
# ## load a MOABB dataset

# %% [markdown]
# ## fit and evaluat the model for all domains

# %%
random_seed=42
torch.manual_seed(random_seed)
records = []
records1=[]
# input space alignment
if cfg['input_align']:
    for i in domain.unique():
        X[domain == i] = fn.euler_align(X[domain == i])
X = torch.from_numpy(X)
y = sklearn.preprocessing.LabelEncoder().fit_transform(labels)
y = torch.from_numpy(y)  


#! 10-fold CV with subjects as groups
subject_count = len(np.unique(subjects))

if subject_count <10:
    cv_outer = sklearn.model_selection.LeaveOneGroupOut()
else:
    cv_outer = sklearn.model_selection.GroupKFold(n_splits=10)
cv_outer_group = subjects
#! train/validation split stratified across domains and labels
cv_inner_group = [f"{d.item()}_{l}" for d, l in zip(domain, y)]
cv_inner_group = sklearn.preprocessing.LabelEncoder().fit_transform(cv_inner_group)
# add datadependen model kwargs
mdl_kwargs = deepcopy(cfg['mdl_kwargs'])
mdl_kwargs['num_classes'] = n_classes
mdl_kwargs['num_electrodes'] = X.shape[1]
mdl_kwargs['chunk_size'] = X.shape[2]
mdl_kwargs['domains'] = domain.unique()
for ix_fold, (fit, test) in enumerate(cv_outer.split(X, y, cv_outer_group)):

    # split fitting data into train and validation 
    cv_inner = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=cfg['validation_size'])
    train, val = next(cv_inner.split(X[fit], y[fit], cv_inner_group[fit]))

    # adjust number of 
    du = domain[fit][train].unique()
    if cfg['domains_per_batch'] > len(du):
        domains_per_batch = len(du)
    else:
        domains_per_batch = cfg['domains_per_batch']

    # split entire dataset into train/validation/test
    ds_train = DomainDataset(X[fit][train], y[fit][train], domain[fit][train])
    ds_val = DomainDataset(X[fit][val], y[fit][val], domain[fit][val])
    

    # create dataloaders
    # for training use specific loader/sampler so taht 
    # batches contain a specific number of domains with equal observations per domain
    # and stratified labels
    loader_train = StratifiedDomainDataLoader(ds_train, cfg['batch_size_train'], domains_per_batch=domains_per_batch, shuffle=True, drop_last = False)
    loader_val = torch.utils.data.DataLoader(ds_val, batch_size=len(ds_val))
    test_domain=domain[test].unique()

    # create the model and training configuration
    model = HEEGNet(device=device,dtype=cfg['dtype'],**mdl_kwargs).to(device=device, dtype=cfg['dtype'])
    # early stopping
    es = EarlyStopping(metric='val_loss', higher_is_better=False, patience=20, verbose=False)
    # batchnorm momentum scheduler
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
    # fit the model
    trainer.fit(model, train_dataloader=loader_train, val_dataloader=loader_val)
    print(f'ES best epoch={es.best_epoch}')
    sfuda_offline_net= deepcopy(model)

    # evaluation
    test_domain=domain[test].unique()
    for test_domain in test_domain:
        fold=ix_fold
        print(f"Fold:{fold}, test domain: {test_domain}")    
        ds_test = DomainDataset(X[test][domain[test] == test_domain], y[test][domain[test] == test_domain], domain[test][domain[test] == test_domain])
        loader_test = torch.utils.data.DataLoader(ds_test, batch_size=len(ds_test))
        sfuda_offline(ds_test, sfuda_offline_net)
        res = trainer.test(sfuda_offline_net, dataloader=loader_test)
        res2 = res
        print(f"HEEGNet, Test results: {res2}")
        records.append(dict(dataset=dataset,fold=fold,domain=test_domain, **res))




# %%
resdf = pd.DataFrame(records)
resdf.groupby(['dataset']).agg(['mean', 'std']).round(3)


