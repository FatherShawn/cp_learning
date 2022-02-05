import math
import numpy as np
import torch as pt
import torchvision.transforms as tt

def scale_down(source: np.ndarray) -> np.ndarray:
    smallest_square = math.ceil(math.sqrt(source.size))
    square = np.reshape()


def nparray2png(source: np.ndarray) -> np.ndarray:
    # The pre-trained densenet expects 224 x 224 images. A survey of
    # our data shows that 172 x 172 would be large enough so there is
    # buffer space for fluctuation.  To prevent image loss we include
    # a lossless resize for data that is larger than the consistent width.
    consistent_width = 50176  # 224 * 224

    data = source.tobytes()
    converted = np.frombuffer(data, dtype=np.ubyte)
    if converted.size <= 50176:
        unused_width = consistent_width - converted.size
        left = unused_width // 2
        right = unused_width - left
        padded = np.pad(converted, (left, right))
        return np.reshape(padded, (224, 224))
    # Upsize the consistent width.
    smallest_square = math.ceil(math.sqrt(converted.size))
    consistent_width = smallest_square * smallest_square
    # Pad to the smallest square.
    unused_width = consistent_width - converted.size
    left = unused_width // 2
    right = unused_width - left
    padded = np.pad(converted, (left, right))
    # Reshape to a square.
    squared = np.reshape(padded, (smallest_square, smallest_square))
    # Resize needs a tensor.
    square_tensor = pt.from_numpy(squared)
    resize = tt.Resize(size=(224, 224), interpolation=tt.InterpolationMode.NEAREST)
    scaled = resize(square_tensor)
    # Return a numpy array.
    return scaled.numpy()