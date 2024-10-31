import numpy as np
from scipy import linalg

class Image:

    yuv_from_rgb_mat = np.array(
        [
            [0.299, 0.587, 0.114],
            [-0.14714119, -0.28886916, 0.43601035],
            [0.61497538, -0.51496512, -0.10001026],
        ]
    )

    rgb_from_yuv_mat = linalg.inv(yuv_from_rgb_mat)

    def yuv_from_rgb(rgb):     
        return rgb @ Image.yuv_from_rgb_mat.T
    
    def rgb_from_yuv(yuv):
        return yuv @ Image.rgb_from_yuv_mat.T


rgb = np.array([100, 54, 206])
yuv = Image.yuv_from_rgb(rgb)
rgb2 = Image.rgb_from_yuv(yuv)
print(yuv)
print(rgb2)