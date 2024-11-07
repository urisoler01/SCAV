import pywt

class DWT:
    @staticmethod
    def encode(data):
        pywt.dwt2(data, 'haar')

    def decode(coeffs):
        return pywt.waverec(coeffs, 'haar')