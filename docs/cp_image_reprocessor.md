# cp_image_reprocessor module

Iterates through a QuackIterableDataset and creates a QuackImageDataset.


### cp_image_reprocessor.main()
The reprocessing logic.

**Required** arguments are:

> –source_path

>     *str* **Required** The path to top dir of the QuackIterableDataset.

> –storage_path

>     *str* **Required** The top directory of the data storage tree for the QuackImageDataset.

**Optional** arguments are:

    \` –filtered\`

        > *bool* Flag to only include censored and uncensored data.

        –undetermined

            *bool* Flag to include only undetermined data

        –start

            *int* The starting index in the QuackIterableDataset.

        –end

            *int* The ending index in the QuackIterableDataset.


* **Return type**

    void
