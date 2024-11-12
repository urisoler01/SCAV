import pywt
import numpy as np

class DWT:
    @staticmethod
    def encode(data):
        if not isinstance(data, np.ndarray): #Cast always the input to an array type
            data = np.array(data)

        # If the length is odd, pad it to make it even
        data_len = len(data)
        pad_len = 0
        if data_len % 2 != 0:
            pad_len = 1  # We will pad one element if length is odd as DWT needs to have even length input
            data = np.pad(data, (0, pad_len), mode='constant')

        # Perform the DWT and return the coefficients (cA, cD), also indicates if padding has been performed
        return pywt.dwt(data, 'db1'), pad_len

    @staticmethod
    def decode(coeffs, pad_len):
        if not isinstance(coeffs, np.ndarray):
            coeffs = np.array(coeffs)
            
        # Use the inverse DWT to reconstruct the data
        reconstructed = pywt.idwt(coeffs[0], coeffs[1], 'db1')
    
        # If padding was applied, trim it from the reconstructed data
        if pad_len > 0:
            return reconstructed[:-pad_len]
        return reconstructed