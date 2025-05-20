
<p align="center">
  <img src="jaxflow/resources/logo.png" alt="JAXFlow Logo" width="300"/>
</p>

[![PyPI version](https://img.shields.io/pypi/v/jaxflow)](https://pypi.org/project/jaxflow/)
[![License](https://img.shields.io/pypi/l/jaxflow)](https://github.com/mthd98/JAXFlow/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/mthd98/JAXFlow/ci.yml?branch=main)](https://github.com/mthd98/JAXFlow/actions)
[![Coverage Status](https://img.shields.io/codecov/c/github/mthd98/JAXFlow)](https://codecov.io/gh/mthd98/JAXFlow)



# JAXFlow

A lightweight neural-network library built on [JAX](https://github.com/google/jax)
– pure-functional, multi-device-ready, and flexible enough for both research and production.

---

## 🚀 Features

> Built from scratch with ❤️ and powered by [JAX](https://github.com/google/jax), JAXFlow began as a deep dive into how libraries like Keras and scikit-learn work under the hood—and evolved into a full-featured framework for high-performance deep learning and machine learning.

* **Modular Model API**
  Build networks using `Sequential`, subclassed `Model`s, or pure-layer stacks.
* **Multi-Device Execution**
  Fully compatible with `jit`, `vmap`, `pmap`, and `pjit` via PyTree-aware design.
* **Layer Collection**
  `Dense`, `Conv`, `BatchNorm`, `Dropout`, `Flatten`, `Embedding`, and custom `Layer` subclasses.
* **Train-Eval Pipelines**
  `model.compile()` + `fit()` for simplicity, or write your own training loop for advanced control.
* **Optimizers & Schedulers**
  Integrated with [Optax](https://github.com/deepmind/optax), supports SGD, Adam, RMSProp, and more.
* **Losses & Metrics**
  MSE, CrossEntropy, F1Score, Precision, Recall, Accuracy, etc. via streaming metric classes.
* **Callbacks & Checkpoints**
  EarlyStopping, ModelCheckpoint, LearningRateScheduler, and Orbax-powered save/load.
* **Pre-built Models**
  Includes `ResNet`, `MLP`, `Transformer`, and composable `Block`s.
* **Lazy Imports**
  Top-level `jaxflow` is fast to import; deep components load on demand.

---

## 📦 Installation

```bash
pip install jaxflow
```

> Note:
>
> Requires JAX with CPU/GPU/TPU support.
>
> ```bash
> pip install "jax[cuda]>=0.6.0" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
> ```
>
> Or simply use:
>
> ```bash
> pip install --upgrade jaxflow[GPU]   # for CUDA support
> pip install --upgrade jaxflow[tpu]   # for TPU support
> ```

Python ≥3.9 required.

---

## 🎉 Quickstart

JAXFlow models can be defined in two main styles:

### 1. Subclassing `Model`

```python
import jaxflow as jf
from jaxflow.models import Model
from jaxflow.layers import Conv2D, MaxPooling2D, Dense
from jaxflow.initializers import GlorotUniform, Zeros

class CNN(Model):
    def __init__(self, num_classes: int = 10, name: str = "MyCNN"):
        super().__init__(name=name)
        self.conv1 = Conv2D(filters=32, kernel_size=(3,3), activation=jf.activations.relu, kernel_initializer=GlorotUniform, bias_initializer=Zeros, padding='SAME')
        self.pool1 = MaxPooling2D(pool_size=(2,2))
        self.conv2 = Conv2D(filters=64, kernel_size=(3,3), activation=jf.activations.relu, kernel_initializer=GlorotUniform, bias_initializer=Zeros, padding='SAME')
        self.pool2 = MaxPooling2D(pool_size=(2,2))
        self.flatten = jf.layers.GlobalAveragePooling2D()
        self.dense1 = Dense(units=64, activation=jf.activations.relu, kernel_initializer=GlorotUniform, bias_initializer=Zeros)
        self.outputs = Dense(units=num_classes, activation=jf.activations.softmax, kernel_initializer=GlorotUniform, bias_initializer=Zeros)

    def call(self, inputs, training: bool = False):
        x = self.conv1(inputs, training=training)
        x = self.pool1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.pool2(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x, training=training)
        return self.outputs(x, training=training)
```

### 2. Using the `.add()` Method (Sequential-style API)

```python
import jaxflow as jf
from jaxflow.models import Model
from jaxflow.layers import Conv2D, MaxPooling2D, Dense
from jaxflow.initializers import GlorotUniform, Zeros
from jaxflow.optimizers import Adam
from jaxflow.losses import SparseCategoricalCrossentropy

model = Model()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation=jf.activations.relu, kernel_initializer=GlorotUniform, bias_initializer=Zeros, padding='SAME'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation=jf.activations.relu, kernel_initializer=GlorotUniform, bias_initializer=Zeros, padding='SAME'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(jf.layers.GlobalAveragePooling2D())
model.add(Dense(units=64, activation=jf.activations.relu, kernel_initializer=GlorotUniform, bias_initializer=Zeros))
model.add(Dense(units=10, activation=jf.activations.softmax, kernel_initializer=GlorotUniform, bias_initializer=Zeros))

model.build(input_shape=(None, 28, 28, 1))
model.compile(optimizer=Adam(0.001), loss_fn=SparseCategoricalCrossentropy())
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
```

---

## 📖 Documentation

Whether you're exploring JAX, need scalable training tools, or just love building things—check it out and let us know what you think!

* 🔗 GitHub: [github.com/mthd98/JAXFlow](https://github.com/mthd98/JAXFlow)

* 📦 PyPI: [pypi.org/project/jaxflow](https://pypi.org/project/jaxflow/)

* 📘 [API Reference](https://mthd98.github.io/JAXFlow/)


---

## 🛠️ Structure

```
jaxflow/
├── core/           # Variable management, RNG scopes
├── gradient/       # Autograd and custom gradients
├── activations/    # relu, gelu, swiglu, ...
├── initializers/   # he_normal, glorot_uniform, ...
├── layers/         # Conv2D, Dense, LayerNorm, ...
├── losses/         # mse, cross_entropy, ...
├── optimizers/     # Optax integration
├── callbacks/      # EarlyStopping, Logger, Checkpointing
├── metrics/        # Precision, Recall, Accuracy, ...
├── models/         # Sequential, ResNet, Transformer
└── regularizers/   # Dropout, L2, ...
```

---

## 🚧 Coming Soon

* Transformer layer with attention
* Callback system (EarlyStopping, ModelCheckpoint, etc.)
* Model saving/loading
* Classical ML models (SVM, Logistic Regression, KNN, Random Forest)
---

## 📄 License

JAXFlow is distributed under the Apache-2.0 License. See `LICENSE` for full details.

---

> With JAXFlow, keep your research code clean, fast, and scalable.
