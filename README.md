
<p align="center">
  <img src="jaxflow/resources/logo.png" alt="JAXFlow Logo" width="300"/>
</p>

[![PyPI version](https://img.shields.io/pypi/v/jaxflow)](https://pypi.org/project/jaxflow/)
[![License](https://img.shields.io/pypi/l/jaxflow)](https://github.com/mthd98/JAXFlow/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/mthd98/JAXFlow/ci.yml?branch=main)](https://github.com/mthd98/JAXFlow/actions)
[![Coverage Status](https://img.shields.io/codecov/c/github/mthd98/JAXFlow)](https://codecov.io/gh/mthd98/JAXFlow)


---

# JAXFlow: A JAX-based Deep Learning and Machine Learning  Framework

**JAXFlow** is a modern, lightweight neural network library built on top of [JAX](https://github.com/google/jax). It is a **pure-functional**, multi-device-ready, and modular **deep learning framework** designed for research, experimentation, and production-ready machine learning pipelines.

> If you're searching for a fast, flexible, and fully-JAX-compatible framework for building neural networks, JAXFlow is designed for you.

---

## 🚀 Why JAXFlow?

> JAXFlow is not just another wrapper around JAX—it's a ground-up, PyTree-aware system for creating, training, and deploying high-performance deep learning models with minimal overhead and full control.

### 🔑 Key Features

* ✅ **Modular Model API**
  Define networks using `Sequential`, subclassed `Model`s, or flexible functional blocks.

* 🦮 **JAX-Compatible Execution**
  Built from the ground up to support `jit`, `vmap`, `pmap`, `pjit`, and full PyTree semantics.

* 🗺 **Rich Layer Library**
  Includes `Dense`, `Conv`, `BatchNorm`, `Embedding`, `Dropout`, and custom `Layer` classes.

* 🏋️ **Training API**
  Use `.compile()` + `.fit()` or write custom training loops for full control.

* ⚙️ **Optimizers & Schedulers**
  Built-in integration with [Optax](https://github.com/deepmind/optax).

* 📊 **Losses & Streaming Metrics**
  Includes `CrossEntropy`, `MSE`, `Accuracy`, `F1`, `Precision`, and more.

* 📂 **Callbacks & Checkpoints**
  Support for EarlyStopping, LearningRateScheduler, and Orbax save/load.

* 🧠 **Built-in Models**
  Comes with ready-to-use `ResNet`, `MLP`, `Transformer`, and composable blocks.

* ⚡ **Lazy Imports**
  Fast to import, loading deep components only when needed.

---

## 📦 Installation

```bash
pip install jaxflow
```

Requires Python ≥3.9 and a valid JAX installation.

To install JAX with GPU or TPU support:

```bash
pip install "jax[cuda]>=0.6.0" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Or use:

```bash
pip install --upgrade jaxflow[GPU]
pip install --upgrade jaxflow[TPU]
```

---

## 🧑‍💻 Quickstart: Build Your First JAXFlow Model

JAXFlow supports two modeling styles: subclassing and sequential-style.

### 1. Subclassing a Model

```python
import jaxflow as jf
from jaxflow.models import Model
from jaxflow.layers import Conv2D, MaxPooling2D, Dense
from jaxflow.initializers import GlorotUniform, Zeros

class CNN(Model):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = Conv2D(32, (3, 3), activation=jf.activations.relu)
        self.pool1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (3, 3), activation=jf.activations.relu)
        self.pool2 = MaxPooling2D((2, 2))
        self.flatten = jf.layers.GlobalAveragePooling2D()
        self.dense1 = Dense(64, activation=jf.activations.relu)
        self.outputs = Dense(num_classes, activation=jf.activations.softmax)

    def call(self, inputs, training=False):
        x = self.conv1(inputs, training=training)
        x = self.pool1(x)
        x = self.conv2(x, training=training)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x, training=training)
        return self.outputs(x, training=training)
```

### 2. Sequential API with .add()

```python
model = jf.models.Model()
model.add(jf.layers.Conv2D(32, (3, 3), activation=jf.activations.relu))
model.add(jf.layers.MaxPooling2D((2, 2)))
model.add(jf.layers.Conv2D(64, (3, 3), activation=jf.activations.relu))
model.add(jf.layers.MaxPooling2D((2, 2)))
model.add(jf.layers.GlobalAveragePooling2D())
model.add(jf.layers.Dense(64, activation=jf.activations.relu))
model.add(jf.layers.Dense(10, activation=jf.activations.softmax))

model.build(input_shape=(None, 28, 28, 1))
model.compile(optimizer=jf.optimizers.Adam(0.001), loss_fn=jf.losses.SparseCategoricalCrossentropy())
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
```

---

## 📖 Documentation

Full documentation and API reference available:

* 🌐 [GitHub Repo](https://github.com/mthd98/JAXFlow)
* 📘 [API Docs](https://mthd98.github.io/JAXFlow/)
* 📦 [PyPI Package](https://pypi.org/project/jaxflow/)

---

## 🗂️ Project Structure

```
jaxflow/
├── activations/       # ReLU, GELU, Swish, etc.
├── callbacks/         # EarlyStopping, Logger, Checkpointing
├── core/              # Base module, scopes, tree utilities
├── gradient/          # JAX custom grad support
├── initializers/      # Glorot, He, Zeros, ...
├── layers/            # Dense, Conv2D, Embedding, ...
├── losses/            # CrossEntropy, MSE, ...
├── metrics/           # Accuracy, F1, Precision, ...
├── models/            # Sequential, Transformer, ResNet
├── optimizers/        # Optax integrations
└── regularizers/      # L1, L2, Dropout
```

---

## 🔮 What’s Next

Planned additions to JAXFlow:

* ☑️ Transformer layer with attention support
* ☑️ Full callback system with exportable training logs
* ☑️ Model persistence and loading with Orbax
* ☑️ Classical ML algorithms: SVM, KNN, Logistic Regression

---

## 📄 License

JAXFlow is licensed under the Apache-2.0 License.

💖 Built with care for the JAX community. Keep your models clean, fast, and functional—with JAXFlow.

jaxflow, jax deep learning, jax neural network library, jax model framework, python deep learning, neural network in jax, jaxflow documentation, jaxflow github, functional deep learning, modular jax library
