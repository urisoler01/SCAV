import unittest
import numpy as np
from io import StringIO
import os
from unittest.mock import patch
from scipy import linalg
import subprocess
from Lab2.main import Image as Image
from dwt import DWT as DWT
from dct import DCT as DCT

# Test for Image class
class TestImage(unittest.TestCase):
    
    #Check values in https://www.calculatormix.com/conversions/color/rgb-to-yuv/
    def test_yuv_from_rgb(self):
        rgb = np.array([255, 0, 0])  # Red color in RGB
        expected_yuv = np.array([82, 90, 240])  # Expected YUV for red
        result = Image.yuv_from_rgb(rgb)
        np.testing.assert_almost_equal(result, expected_yuv, decimal=2)

    def test_rgb_from_yuv(self):
        yuv = np.array([145, 54, 34])  # Red color in YUV
        expected_rgb = np.array([0, 255, 0])  # Expected RGB for red
        result = Image.rgb_from_yuv(yuv)
        np.testing.assert_almost_equal(result, expected_rgb, decimal=2)
    

    #This unit test doesn't make sense as we would compare a downsized image probably
    #extracted from ffmpeg to a image already generated with ffmpeg but inside a python file
    @patch('subprocess.run')
    def test_downsize(self, mock_run):
        # Mock the subprocess call to avoid actual file manipulation
        mock_run.return_value = None
        image_file_name = "test_image.jpg"
        Image.downsize(image_file_name, ratio=2)
        mock_run.assert_called_once()

    def test_serpentine(self):
        file = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_result = [1, 2, 4, 7, 5, 3, 6, 8, 9]
        result = Image.serpentine(file)
        self.assertEqual(result, expected_result)

    #This unit test doesn't make sense as we would compare a compressed image probably
    #extracted from ffmpeg to a image already generated with ffmpeg but inside a python file
    @patch('subprocess.run')
    def test_compress_hard(self, mock_run):
        # Mock the subprocess call to avoid actual file manipulation
        mock_run.return_value = None
        image_file_name = "test_image_compress.jpg"
        Image.compress_hard(image_file_name)
        mock_run.assert_called_once()

    def test_run_length_encoding(self):
        bytes_data = [0, 0, 1, 0, 0, 0, 1, 2]
        expected_encoding = [0, 2, 1, 0, 3, 1, 2]
        result = Image.run_length_encoding(bytes_data)
        self.assertEqual(result, expected_encoding)


# Test for DWT class
class TestDWT(unittest.TestCase):

    def test_encode_even_length(self):
        data = np.array([1, 2, 3, 4])
        coeffs, pad_len = DWT.encode(data)
        self.assertEqual(pad_len, 0)
        self.assertEqual(len(coeffs[0]), len(data) // 2)

    def test_encode_odd_length(self):
        data = np.array([1, 2, 3, 4, 5])
        coeffs, pad_len = DWT.encode(data)
        self.assertEqual(pad_len, 1)
        self.assertEqual(len(coeffs[0]), (len(data) + 1) // 2)

    def test_decode_with_padding(self):
        data = np.array([1, 2, 3, 4, 5])
        coeffs, pad_len = DWT.encode(data)
        reconstructed_data = DWT.decode(coeffs, pad_len)
        np.testing.assert_almost_equal(reconstructed_data, data, decimal=2)

    def test_decode_without_padding(self):
        data = np.array([1, 2, 3, 4])
        coeffs, pad_len = DWT.encode(data)
        reconstructed_data = DWT.decode(coeffs, pad_len)
        np.testing.assert_almost_equal(reconstructed_data, data, decimal=2)


# Test for DCT class
class TestDCT(unittest.TestCase):

    def test_encode(self):
        data = np.array([1, 2, 3, 4])
        coeffs = DCT.encode(data)
        self.assertEqual(len(coeffs), len(data))

    def test_decode(self):
        data = np.array([1, 2, 3, 4])
        coeffs = DCT.encode(data)
        reconstructed_data = DCT.decode(coeffs)
        np.testing.assert_almost_equal(reconstructed_data, data, decimal=2)


if __name__ == '__main__':
    unittest.main()
