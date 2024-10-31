import numpy as np
from scipy import linalg
import subprocess

class Image:

    yuv_from_rgb_mat = np.array(
        [
            [0.299, 0.587, 0.114],
            [-0.14714119, -0.28886916, 0.43601035],
            [0.61497538, -0.51496512, -0.10001026],
        ]
    )

    rgb_from_yuv_mat = linalg.inv(yuv_from_rgb_mat)

    @staticmethod
    def yuv_from_rgb(rgb):     
        return rgb @ Image.yuv_from_rgb_mat.T

    @staticmethod
    def rgb_from_yuv(yuv):
        return yuv @ Image.rgb_from_yuv_mat.T

    @staticmethod
    def downsize(image_file_name, ratio = 2):
        command = 'ffmpeg -i {} -vf scale=iw/{}:ih/{} small{} -y'
        command = command.format(image_file_name, ratio, ratio, image_file_name)  # Replace 'image.jpg' with your actual file name.
        subprocess.run(command.split(sep=' '), shell=True)


rgb = np.array([100, 54, 206])
yuv = Image.yuv_from_rgb(rgb)
rgb2 = Image.rgb_from_yuv(yuv)
print(yuv)
print(rgb2)

Image.downsize("1.png", 3)