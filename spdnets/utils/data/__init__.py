import torch
from torch.utils.data import Dataset, Subset, DataLoader, Sampler
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold
import numpy as np

from typing import Iterator


class DomainDataset(Dataset):

    def __init__(self, 
                 features : torch.Tensor,
                 labels : torch.LongTensor,
                 domains : torch.LongTensor):
        self.features = features
        self.domains = domains
        self.labels =  labels

    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self, index):
        return [dict(inputs=self.features[index],domains=self.domains[index]), self.labels[index]]
    

class StratifiedDomainDataLoader(DataLoader):

    def __init__(self, dataset = None, batch_size = 1, domains_per_batch = 1, shuffle=True, **kwargs):

        if isinstance(dataset, Subset) and isinstance(dataset.dataset, Subset) and isinstance(dataset.dataset.dataset, DomainDataset):
            domains = dataset.dataset.dataset.domains[dataset.dataset.indices][dataset.indices]
            labels = dataset.dataset.dataset.labels[dataset.dataset.indices][dataset.indices]
        elif isinstance(dataset, Subset) and isinstance(dataset.dataset, DomainDataset):
            domains = dataset.dataset.domains[dataset.indices]
            labels = dataset.dataset.labels[dataset.indices]
        elif isinstance(dataset, DomainDataset):
            domains = dataset.domains
            labels = dataset.labels
        else:
            raise NotImplementedError()

        sampler = StratifiedDomainSampler(domains, labels,
                int(batch_size/domains_per_batch), domains_per_batch, 
                shuffle=shuffle)

        super().__init__(dataset=dataset, sampler=sampler, batch_size=batch_size, **kwargs)


class StratifiedDomainSampler():

    def __init__(self, domains, stratvar, samples_per_domain, domains_per_batch, shuffle = True) -> None:
        self.samples_per_domain = samples_per_domain
        self.domains_per_batch = domains_per_batch
        self.shuffle = shuffle
        self.stratvar = stratvar

        du, didxs, counts = domains.unique(return_inverse=True, return_counts=True)
        du = du.tolist()
        didxs = didxs.tolist()

        if len(du) < self.domains_per_batch:
            self.domains_per_batch = len(du)
            self.samples_per_domain = int(samples_per_domain * domains_per_batch / self.domains_per_batch)
            print("Warning: fewer domains than domain_per_batch. Adjusted parameter.")
            print(f"domains: {domains_per_batch}->{self.domains_per_batch} samples_per_domain: {samples_per_domain}->{self.samples_per_domain}")

        self.domaincounts = torch.LongTensor((counts/self.samples_per_domain).tolist())
        
        self.domaindict = {}
        for domix, _ in enumerate(du):
            self.domaindict[domix] = torch.LongTensor(
                [idx for idx,dom in enumerate(didxs) if dom == domix])

    def __iter__(self) -> Iterator[int]:

        domaincounts = self.domaincounts.clone()

        generators = {}
        for domain in self.domaindict.keys():
            if self.shuffle:
                permidxs = torch.randperm(self.domaindict[domain].shape[0])
            else:
                permidxs = range(self.domaindict[domain].shape[0])
            generators[domain] = \
                iter(
                    StratifiedSampler(
                        self.stratvar[self.domaindict[domain]], 
                        batch_size=self.samples_per_domain,
                        shuffle=self.shuffle
                    ))

        while domaincounts.sum() > 0:

            assert((domaincounts >= 0).all())
            # candidates = [ix for ix, count in enumerate(domaincounts.tolist()) if count > 0]
            candidates = torch.nonzero(domaincounts, as_tuple=False).flatten()
            if candidates.shape[0] < self.domains_per_batch:
                break

            # candidates = torch.LongTensor(candidates)
            permidxs = torch.randperm(candidates.shape[0])
            candidates = candidates[permidxs]

            # icap = min(len(candidates), self.domains_per_batch)
            batchdomains = candidates[:self.domains_per_batch]
            
            for item in batchdomains.tolist():
                within_domain_idxs = [next(generators[item]) for _ in range(self.samples_per_domain)]
                batch = self.domaindict[item][within_domain_idxs]
                # batch = next(generators[item])
                domaincounts[item] = domaincounts[item] - 1
                yield from batch
        yield from []

    def __len__(self) -> int:
        return self.domaincounts.sum() * self.samples_per_domain 


class StratifiedSampler(Sampler[int]):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, stratvar, batch_size, shuffle = True):
        self.n_splits = max(int(stratvar.shape[0] / batch_size), 2)
        self.stratvar = stratvar
        self.shuffle = shuffle

    def gen_sample_array(self):
        if self.shuffle:
            random_state = torch.randint(0,int(1e8),size=()).item()
        else:
            random_state = None
        s = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=random_state)   
        indices = [test for _, test in s.split(self.stratvar, self.stratvar)]
        return np.hstack(indices)

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.stratvar)
