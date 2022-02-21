# cp_flatten_processor module

A controller script for flattening Censored Planet data.


### cp_flatten_processor.main()
Flattens the data in a single .tar file and adds it to the dataset under construction.

**Required** arguments are:

> > –source_path

> >     *str*   The path to the .tar file.  May be local or a url. Passed to CensoredPlanetFlatten.

> \` –storage_path\`

>     > *str* The top directory of the data storage tree.

>     –log_path

>         *str* default=0 The path to a log file.

>     –vocab_path

>         *str* default=0 The path to a .pyc file.  Passed to CensoredPlanetFlatten.


### cp_flatten_processor.verify_returned_item(item: [cp_flatten.TokenizedQuackData](cp_flatten.md#cp_flatten.TokenizedQuackData))
Utility function to check the structure of a TokenizedQuackData item.


* **Parameters**

    **item** ([*TokenizedQuackData*](cp_flatten.md#cp_flatten.TokenizedQuackData)) – The item to check



* **Raises**

    **AssertionError** – Thrown if any test of the structure fails.



* **Return type**

    void
