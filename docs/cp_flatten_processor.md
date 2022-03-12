# cp_flatten_processor module

A controller script for flattening Censored Planet data.


### cp_flatten_processor.verify_returned_item(item: [cp_flatten.TokenizedQuackData](cp_flatten.md#cp_flatten.TokenizedQuackData))
Utility function to check the structure of a TokenizedQuackData item.


* **Parameters**

    **item** ([*TokenizedQuackData*](cp_flatten.md#cp_flatten.TokenizedQuackData)) – The item to check



* **Raises**

    **AssertionError** – Thrown if any test of the structure fails.



* **Return type**

    void



### cp_flatten_processor.main()
Flattens the data in a single .tar file and adds it to the dataset under construction.

**Required** arguments are:
