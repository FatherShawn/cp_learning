# latent_reprocessor module

Iterates through the output of the prediction loop from QuackAutoencoder and structures the files for use
as a dataset.


### latent_reprocessor.main()
The reprocessing logic.

**Required** arguments are:

> –source_path

>     *str* **Required** The path to top dir of the QuackIterableDataset.

> –storage_path

>     *str* **Required** The top directory of the data storage tree for the QuackImageDataset.


* **Return type**

    void
