# nparray2png module


### nparray2png.nparray2png(source: numpy.ndarray)
Converts a 1D numpy array into a 2D image array.

The pre-trained densenet expects 224 x 224 images. A survey of
our data shows that 172 x 172 would be large enough so there is
buffer space for fluctuation.  To prevent image loss we include
a lossless resize for data that is larger than the consistent width.


* **Parameters**

    **source** (*np.ndarray*) – The source array



* **Returns**

    A (224, 224) matrix of pixels in the range 0..255.



* **Return type**

    np.ndarray
