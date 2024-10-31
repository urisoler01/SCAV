import numpy as np
from scipy import linalg

class Image:
       
    yuv_from_rgb = np.array(
        [
            [0.299, 0.587, 0.114],
            [-0.14714119, -0.28886916, 0.43601035],
            [0.61497538, -0.51496512, -0.10001026],
        ]
    )

    rgb_from_yuv = linalg.inv(yuv_from_rgb)

    def yuv_from_rgb(rgb):     
        return rgb @ yuv_from_rgb.T.astype(arr.dtype)
    def rgb_from_yuv(yuv):
        return yuv @ rgb_from_yuv.T.astype(arr.dtype)

