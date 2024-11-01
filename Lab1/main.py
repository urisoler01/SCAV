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
        # -vf scale= is used to downscale the image iw and ih are image width and height, which get divided by ratio
        command = 'ffmpeg -i {} -vf scale=iw/{}:ih/{} small{} -y'
        command = command.format(image_file_name, ratio, ratio, image_file_name)
        subprocess.run(command.split(sep=' '), shell=True)


    @staticmethod
    def compress_hard(image_file_name):
        # -vf format=gray declares grayscale, maskfun is used to split pixels into only white or black, threshold is 128
        # -qscale:v 31 applies the maximum jpg compression and -preset slow puts quality above execution speed
        command = 'ffmpeg -i {} -vf format=gray,maskfun=low=128:high=128:fill=0:sum=128 -qscale:v 31 -preset slow compressed{} -y'
        command = command.format(image_file_name, "".join(image_file_name.split('.')[:-1]) + '.jpg')
        subprocess.run(command.split(sep=' '), shell=True)




rgb = np.array([100, 54, 206])
yuv = Image.yuv_from_rgb(rgb)
rgb2 = Image.rgb_from_yuv(yuv)
print(yuv)
print(rgb2)

Image.downsize("img.png", 3)

Image.compress_hard("img.png")