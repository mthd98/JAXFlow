<p align="center">
  <img src="jaxflow/resources/logo.png" alt="JAXFlow Logo" width="300"/>
</p>



# JAXFlow

[![PyPI version](https://img.shields.io/pypi/v/jaxflow)](https://pypi.org/project/jaxflow/) 
[![License](https://img.shields.io/pypi/l/jaxflow)](https://github.com/mthd98/JAXFlow/blob/main/LICENSE) 
[![Build Status](https://img.shields.io/github/actions/workflow/status/mthd98/JAXFlow/ci.yml?branch=main)](https://github.com/mthd98/JAXFlow/actions) 
[![Coverage Status](https://img.shields.io/codecov/c/github/mthd98/JAXFlow)](https://codecov.io/gh/mthd98/JAXFlow)

A lightweight neural-network library built on [JAX](https://github.com/google/jax)  
– fast imports, pure-functional APIs, and batteries-included for research and production.

---

## 🚀 Features

- **Module API**  
  Define layers with familiar `setup`/`__call__` style or pure-function transforms.
- **PyTree compatibility**  
  Everything is a JAX PyTree; seamless `jit`, `vmap`, `pmap`, and `pjit`.
- **Rich layer collection**  
  `Dense`, `Conv`, `BatchNorm`, `Dropout`, and more in `jaxflow.layers`.
- **Optimizers & Schedulers**  
  Thin wrappers around [Optax](https://github.com/deepmind/optax) in `jaxflow.optimizers`.
- **Activations & Initializers**  
  `relu`, `gelu`, `swish`, `he_normal`, `glorot_uniform`, …
- **Losses & Metrics**  
  Standard losses (`mse`, `cross_entropy`) and metrics (`accuracy`, `precision`, …).
- **Callbacks & Checkpointing**  
  Training hooks and Orbax-powered `jaxflow.checkpt` utilities.
- **Pre-built Models**  
  `ResNet`, `Transformer`, `MLP`, and easy to extend in `jaxflow.models`.
- **Modular & Lazy**  
  Top-level import is lightning-fast; submodules load on demand.

---

## 📦 Installation
bash
pip install jaxflow

> Note:

Requires JAX (CPU/GPU/TPU), e.g.

pip install "jax[cuda]>=0.4.0" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Python ≥3.8





---

🎉 Quickstart



---

📖 Documentation

API Reference: https://mthd98.github.io/JAXFlow/

Cookbook: Layer recipes, advanced transforms, multi-host training



---

🛠️ Structure

jaxflow/
├── core/           # Variable, RNG contexts
├── gradient/       # Gradient utilities
├── activations/    # relu, gelu, swish, …
├── initializers/   # weight initializers
├── layers/         # Dense, Conv, BatchNorm, …
├── losses/         # mse, cross_entropy, …
├── optimizers/     # Optax wrappers, schedulers
├── callbacks/      # EarlyStopping, Logging, …
├── metrics/        # accuracy, precision, …
├── models/         # ResNet, Transformer, …
└── regularizers/   # Dropout, weight decay, …


---

🤝 Contributing

We welcome contributions! Please see our CONTRIBUTING.md for:

1. Setting up a dev environment


2. Code style & linting


3. Testing & CI guidelines


4. How to file issues & propose features




---

📄 License

This project is licensed under the Apache-2.0 License. See the LICENSE file for details.


---

> “Simplicity is the ultimate sophistication.” – Leonardo da Vinci
With JAXFlow, keep your research code clean, fast, and reproducible.





