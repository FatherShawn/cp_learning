# blockpage module

Matcher for response pages to blockpage signatures.

### References

Adapted from:
[https://github.com/censoredplanet/censoredplanet-analysis](https://github.com/censoredplanet/censoredplanet-analysis)


### _class_ blockpage.BlockpageMatcher()
Bases: `object`

Matcher to confirm blockpages or false positives.

### References

Adapted from:
[https://github.com/censoredplanet/censoredplanet-analysis/blob/master/pipeline/metadata/blockpage.py](https://github.com/censoredplanet/censoredplanet-analysis/blob/master/pipeline/metadata/blockpage.py)


#### match_page(page: str)
Check if the input page matches a known blockpage or false positive.


* **Parameters**

    **page** (*str*) – A string containing the HTTP body of the potential blockpage



* **Returns**

    (match_outcome, match_fingerprint)

    match_outcome is

    > 
    > * *True* if page matches a blockpage signature.


    > * *False* if page matches a false positive signature.


    > * *None* otherwise.

    match_fingerprint is a signature for a blockpage/fp like ‘a_prod_cisco’




* **Return type**

    Tuple[Optional[bool], Optional[str]]
