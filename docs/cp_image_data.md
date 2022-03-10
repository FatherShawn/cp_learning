# cp_image_data module


### _class_ cp_image_data.QuackImageDataModule(\*args: Any, \*\*kwargs: Any)
Bases: `pytorch_lightning.core.datamodule.LightningDataModule`


#### predict_dataloader()
Implement one or multiple PyTorch DataLoaders for prediction.

It’s recommended that all data downloads and preparation happen in `prepare_data()`.


* `fit()`


* …


* `prepare_data()`


* `train_dataloader()`


* `val_dataloader()`


* `test_dataloader()`

**NOTE**: Lightning adds the correct sampler for distributed and arbitrary hardware
There is no need to set it yourself.


* **Returns**

    A `torch.utils.data.DataLoader` or a sequence of them specifying prediction samples.


**NOTE**: In the case where you return multiple prediction dataloaders, the `predict()`
will have an argument `dataloader_idx` which matches the order here.


#### test_dataloader()
Implement one or multiple PyTorch DataLoaders for testing.

The dataloader you return will not be reloaded unless you set


```
:paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_n_epochs`
```

 to
a postive integer.

For data processing use the following pattern:

> 
> * download in `prepare_data()`


> * process and split in `setup()`

However, the above are only necessary for distributed processing.

**WARNING**: do not assign state in prepare_data


* `fit()`


* …


* `prepare_data()`


* `setup()`


* `train_dataloader()`


* `val_dataloader()`


* `test_dataloader()`

**NOTE**: Lightning adds the correct sampler for distributed and arbitrary hardware.
There is no need to set it yourself.


* **Returns**

    A `torch.utils.data.DataLoader` or a sequence of them specifying testing samples.


Example:

```default
def test_dataloader(self):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (1.0,))])
    dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform,
                    download=True)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=self.batch_size,
        shuffle=False
    )

    return loader

# can also return multiple dataloaders
def test_dataloader(self):
    return [loader_a, loader_b, ..., loader_n]
```

**NOTE**: If you don’t need a test dataset and a `test_step()`, you don’t need to implement
this method.

**NOTE**: In the case where you return multiple test dataloaders, the `test_step()`
will have an argument `dataloader_idx` which matches the order here.


#### train_dataloader()
Implement one or more PyTorch DataLoaders for training.


* **Returns**

    A collection of `torch.utils.data.DataLoader` specifying training samples.
    In the case of multiple dataloaders, please see this page.


The dataloader you return will not be reloaded unless you set


```
:paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_n_epochs`
```

 to
a positive integer.

For data processing use the following pattern:

> 
> * download in `prepare_data()`


> * process and split in `setup()`

However, the above are only necessary for distributed processing.

**WARNING**: do not assign state in prepare_data


* `fit()`


* …


* `prepare_data()`


* `setup()`


* `train_dataloader()`

**NOTE**: Lightning adds the correct sampler for distributed and arbitrary hardware.
There is no need to set it yourself.

Example:

```default
# single dataloader
def train_dataloader(self):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (1.0,))])
    dataset = MNIST(root='/path/to/mnist/', train=True, transform=transform,
                    download=True)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=self.batch_size,
        shuffle=True
    )
    return loader

# multiple dataloaders, return as list
def train_dataloader(self):
    mnist = MNIST(...)
    cifar = CIFAR(...)
    mnist_loader = torch.utils.data.DataLoader(
        dataset=mnist, batch_size=self.batch_size, shuffle=True
    )
    cifar_loader = torch.utils.data.DataLoader(
        dataset=cifar, batch_size=self.batch_size, shuffle=True
    )
    # each batch will be a list of tensors: [batch_mnist, batch_cifar]
    return [mnist_loader, cifar_loader]

# multiple dataloader, return as dict
def train_dataloader(self):
    mnist = MNIST(...)
    cifar = CIFAR(...)
    mnist_loader = torch.utils.data.DataLoader(
        dataset=mnist, batch_size=self.batch_size, shuffle=True
    )
    cifar_loader = torch.utils.data.DataLoader(
        dataset=cifar, batch_size=self.batch_size, shuffle=True
    )
    # each batch will be a dict of tensors: {'mnist': batch_mnist, 'cifar': batch_cifar}
    return {'mnist': mnist_loader, 'cifar': cifar_loader}
```


#### val_dataloader()
Implement one or multiple PyTorch DataLoaders for validation.

The dataloader you return will not be reloaded unless you set


```
:paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_n_epochs`
```

 to
a positive integer.

It’s recommended that all data downloads and preparation happen in `prepare_data()`.


* `fit()`


* …


* `prepare_data()`


* `train_dataloader()`


* `val_dataloader()`


* `test_dataloader()`

**NOTE**: Lightning adds the correct sampler for distributed and arbitrary hardware
There is no need to set it yourself.


* **Returns**

    A `torch.utils.data.DataLoader` or a sequence of them specifying validation samples.


Examples:

```default
def val_dataloader(self):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (1.0,))])
    dataset = MNIST(root='/path/to/mnist/', train=False,
                    transform=transform, download=True)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=self.batch_size,
        shuffle=False
    )

    return loader

# can also return multiple dataloaders
def val_dataloader(self):
    return [loader_a, loader_b, ..., loader_n]
```

**NOTE**: If you don’t need a validation dataset and a `validation_step()`, you don’t need to
implement this method.

**NOTE**: In the case where you return multiple validation dataloaders, the `validation_step()`
will have an argument `dataloader_idx` which matches the order here.


### _class_ cp_image_data.QuackImageTransformer(step: str, strategy: str)
Bases: `object`
