# Hyperbolic Sliced-Wasserstein via Geodesic and Horospherical Projections

This repository constains the code and the expriments of the paper [Hyperbolic Sliced-Wasserstein via Geodesic and Horospherical Projections](). We propose in this paper two new sliced-Wasserstein discrepancies on hyperbolic spaces by using geodesic and horospherical projections. We compare these discrepancies on different tasks such as gradient flows and classification.

## Abstract 

It has been shown beneficial for many types of data which present an underlying hierarchical structure to be embedded in hyperbolic spaces. Consequently, many tools of machine learning were extended to such spaces, but only few discrepancies to compare probability distributions defined over those spaces exist. Among the possible candidates, optimal transport distances are well defined on such Riemannian manifolds and enjoy strong theoretical properties, but suffer from high computational cost. On Euclidean spaces, sliced-Wasserstein distances, which leverage a closed-form of the Wasserstein distance in one dimension, are more computationally efficient, but are not readily available on hyperbolic spaces. In this work, we propose to derive novel hyperbolic sliced-Wasserstein discrepancies. These constructions use projections on the underlying geodesics either along horospheres or geodesics. We study and compare them on different tasks where hyperbolic representations are relevant, such as sampling or image classification.

## Citation

```
@inproceedings{bonet2023hyperbolic,
  title={Hyperbolic Sliced-Wasserstein via Geodesic and Horospherical Projections},
  author={Bonet, Cl{\'e}ment and Chapel, Laetitia and Drumetz, Lucas and Courty, Nicolas},
  booktitle={Topological, Algebraic and Geometric Learning Workshops 2023},
  pages={334--370},
  year={2023},
  organization={PMLR}
}
```

## Experiments

- In the folder "Runtime", you can find the code to reproduce the experiment of Section 4.
- In the notebook "Evolution_along_WND", we plot the evolution of different discrepancies between wrapped normal distributions (Experiment of Section 5.1).
- In the "Gradient Flows" folder, we report the code to reproduce experiments of Section 5.2.
- In the "Busemann Learning" folder, you can find the code to reproduce the experiments of classification with prototypes reported in Section 5.3.



## Credits

- Wrapped Normal Distribution with Pytorch [Normalizing Flows for Hyperbolic Spaces and Beyond!](https://github.com/joeybose/HyperbolicNF)
- Busemann learning: the code was mainly taken from the original repository [Hyperbolic Busemann Learning with Ideal Prototypes](https://github.com/MinaGhadimiAtigh/Hyperbolic-Busemann-Learning). See also the ECCV22 Tutorial at [Hyperbolic Representation Learning for Computer Vision](https://sites.google.com/view/hyperbolic-tutorial-eccv22/).
