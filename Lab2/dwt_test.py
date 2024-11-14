import sys
import os
import numpy as np

# Add the directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Lab1.dwt import DWT

def test_dwt_instance_method():
    input_data = [1, 2, 3, 4, 5]
    encoded_data, pad_len = DWT.encode(input_data)
    print("Encoded data:", encoded_data)
    decoded_data = DWT.decode(encoded_data, pad_len)
    print("Decoded data:", decoded_data)
    
    assert np.allclose(decoded_data, input_data), "The decoded data does not match the input data."