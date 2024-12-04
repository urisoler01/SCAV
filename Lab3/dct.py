from scipy.fftpack import dct, idct
import numpy as np

class DCT:
    @staticmethod
    def encode(data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        return dct(data, norm='ortho')  # Return DCT coefficients


    @staticmethod
    def decode(coeffs):
        if not isinstance(coeffs, np.ndarray):
            coeffs = np.array(coeffs)
        return idct(coeffs, norm='ortho')  # Perform inverse DCT