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


#### \__init__()
Create a Blockpage Matcher.


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



#### \__dict__(_ = mappingproxy({'__module__': 'blockpage', '__doc__': '\\n  Matcher to confirm blockpages or false positives.\\n\\n  References\\n  ----------\\n  Adapted from:\\n  https://github.com/censoredplanet/censoredplanet-analysis/blob/master/pipeline/metadata/blockpage.py\\n  ', '__init__': <function BlockpageMatcher.__init__>, 'match_page': <function BlockpageMatcher.match_page>, '__dict__': <attribute '__dict__' of 'BlockpageMatcher' objects>, '__weakref__': <attribute '__weakref__' of 'BlockpageMatcher' objects>, '__annotations__': {}}_ )

#### \__module__(_ = 'blockpage_ )

#### \__weakref__()
list of weak references to the object (if defined)
