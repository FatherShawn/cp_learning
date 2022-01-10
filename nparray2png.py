import numpy as np

def nparray2png(source: np.ndarray) -> np.ndarray:
    consistent_width = 29584  # 172 x 172

    data = source.tobytes()
    converted = np.frombuffer(data, dtype=np.ubyte)
    unused_width = consistent_width - converted.size
    left = unused_width // 2
    right = unused_width - left
    padded = np.pad(converted, (left, right))
    return np.reshape(padded, (172, 172))
