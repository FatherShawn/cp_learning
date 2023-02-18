# Censored Planet Processes and Models for Machine Learning

## Overview
This repository contains the code written to support the research for my thesis 
_Finding Latent Features in Internet Censorship Data_. The thesis was further refined and subsequently
published as [Detecting Network-based Internet Censorship via Latent Feature Representation Learning](https://www.sciencedirect.com/science/article/pii/S0167404823000482)
and a preprint is available at [arxiv.org](https://arxiv.org/abs/2209.05152)

The machine learning models are build with [Pytorch](https://pytorch.org) extended by
[PytorchLightning](https://www.pytorchlightning.ai).  Logging was set up to use
[Comet](https://www.comet.com/site/products/ml-experiment-tracking/).
If you wish to use a different [logger](https://pytorch-lightning.readthedocs.io/en/latest/api_references.html#loggers)
that can easily be swapped in your instance of this code.

## Process Diagrams

### Building Datasets

The Censored Planet data needs to be transformed into datasets that can be used with our models. I built my base dataset
by ingesting one large `CP_Quack-echo-YYYY-MM-DD-HH-MM-SS.tar` file at a time to accommodate the speed and stability of
my computing environment.

flowchart TD
    A[/Quack tar file/] -->B(cp_flatten_processor.py)
    B --> C[/Pickled Dictionary<br>Stored at indexed path/]
    C -- iterate --> B
    B --> D[/Create or update<br>metadata.pyc/]
    D --> E(Single tar file processed)

The flattened and vectorized data is stored as pickled dictionaries using an indexed directory structure under the
specified output directory

flowchart TD
    A[Dataset dir] --- 0
    A --- 1
    A --- 2
    A --- B[...]
    A --- m
    A --- i[/metadata.pyc/]
    2 --- 2-0[0]
    2 --- 2-1[1]
    2 --- 2-2[2]
    2 --- 2-c[...]
    2 --- 2-99[99]
    2-2 --- 220[/202000.pyc/]
    2-2 --- 221[/202001.pyc/]
    2-2 --- 222[/202002.pyc/]
    2-2 --- 22c[/.../]
    2-2 --- 229[/202999.pyc/]

These dictionary files are used in the remainder of the project via `QuackIterableDataset` found in
`cp_dataset.py`.  This iterable dataset is managed using `QuackTokenizedDataModule`.

For the image based model, this data is accessed via `QuackTokenizedDataModule` and stored in a two new datasets by
`cp_image_reprocessor.py` using a similar directory tree in which each leaf directory stores a PNG image file and a
pickle file of the encoded pixels and metadata. The first image dataset is balanced between censored and uncensored for
training the replacement classifier layer in DenseNet.  The second set are all the _undetermined_ records.

### Building Embeddings

The flattened and tokenized data is used to train the autoencoder

flowchart TD
    A[QuackIterableDataset] --> B[QuackTokenizedDataModule]
    B --> C(ae_processor.py)
    C --iterate--> B
    C --> D[trained QuackAutoEncoder]

The trained autoencoder model is captured and used as in additional input to `ae_processor.py` to process the data
into two sets of embeddings.  One set is labeled and balanced between censored and uncensored for training the
classifier. The second set are embeddings of the _undetermined_ records.

flowchart TD
    J[/trained QuackAutoEncoder/] --> M
    K[QuackIterableDataset] --> L[QuackTokenizedDataModule]
    L --> M
    M(ae_processor.py) --> N[AutoencoderWriter]
    N --> O[/.pyc file in indexed directory/]

These two datasets of embeddings are managed with `QuackLatentDataModule`.

## Job Scripts

Our data processed in the [CUNY HPCC](https://www.csi.cuny.edu/academics-and-research/research-centers/cuny-high-performance-computing-center)
which uses SLURM to manage jobs. Figuring out how to configure for SLURM was a challenge.  An additional challenge
was that Pytorch no longer supported the older GPUs we had available, so we needed to train in parallel on CPU. I 
eventually solved parallel processing on that architecture by using the [Ray parallel plugin](https://github.com/ray-project/ray_lightning).
These job scripts also contain setup for this plugin.  I've left them here as I had trouble finding examples.  Your
computing environment is almost certainly different and that will cause further changes in your instance of this code.

## Documentation

This documentation is presented in markdown that
was generated from the docstrings within each python module.
It may be found in the `docs` directory here in the repository.  

- [ae_processor](https://github.com/FatherShawn/cp_learning/blob/main/docs/ae_processor.md)
- [autoencoder](https://github.com/FatherShawn/cp_learning/blob/main/docs/autoencoder.md)
- [blockpage](https://github.com/FatherShawn/cp_learning/blob/main/docs/blockpage.md)
- [check_meta](https://github.com/FatherShawn/cp_learning/blob/main/docs/check_meta.md)
- [cp_dataset](https://github.com/FatherShawn/cp_learning/blob/main/docs/cp_dataset.md)
- [cp_flatten](https://github.com/FatherShawn/cp_learning/blob/main/docs/cp_flatten.md)
- [cp_flatten_processor](https://github.com/FatherShawn/cp_learning/blob/main/docs/cp_flatten_processor.md)
- [cp_image_data](https://github.com/FatherShawn/cp_learning/blob/main/docs/cp_image_data.md)
- [cp_image_dataset](https://github.com/FatherShawn/cp_learning/blob/main/docs/cp_image_dataset.md)
- [cp_image_reprocessor](https://github.com/FatherShawn/cp_learning/blob/main/docs/cp_image_reprocessor.md)
- [cp_latent_classifier](https://github.com/FatherShawn/cp_learning/blob/main/docs/cp_latent_classifier.md)
- [cp_tokenized_data](https://github.com/FatherShawn/cp_learning/blob/main/docs/cp_tokenized_data.md)
- [densenet](https://github.com/FatherShawn/cp_learning/blob/main/docs/densenet.md)
- [dn_processor](https://github.com/FatherShawn/cp_learning/blob/main/docs/dn_processor.md)
- [nparray2png](https://github.com/FatherShawn/cp_learning/blob/main/docs/nparray2png.md)