import numpy as np
from scipy import linalg
import subprocess
import os
import dct_test
import dwt_test

class Image:
    yuv_from_rgb_mat = (1/255) * np.array(
        [
            [66.5, 129, 25],
            [-38, -74, 112],
            [112, -94, -18],
        ]
    )

    rgb_from_yuv_mat = np.linalg.inv(yuv_from_rgb_mat)

    # Offset arrays for the YUV and RGB conversions
    yuv_offset = np.array([16, 128, 128])

    @staticmethod
    def yuv_from_rgb(rgb):     
        yuv = (rgb @ Image.yuv_from_rgb_mat.T) + Image.yuv_offset
        return yuv.astype(int)

    @staticmethod
    def rgb_from_yuv(yuv):
        rgb = (yuv - Image.yuv_offset) @ Image.rgb_from_yuv_mat.T
        return rgb.astype(int)
    
    @staticmethod
    def downsize(image_file_name, ratio = 2):
        # Get the absolute path based on the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__)) #we mus include the absolute path to work with any computer
        abs_image_path = os.path.join(script_dir, image_file_name)

        # Check if the file exists
        if not os.path.isfile(abs_image_path):
            print(f"Error: The file '{image_file_name}' does not exist.")
            return
        
        # Prepare the output path in the same directory as the input image
        output_file_name = os.path.join(
            script_dir,
            "".join(image_file_name.split('.')[:-1]) + '_small.jpg'
        )
        
        # -vf scale= is used to downscale the image; iw and ih are image width and height, divided by the ratio
        command = 'ffmpeg -i {} -vf scale=iw/{}:ih/{} {} -y'
        command = command.format(abs_image_path, ratio, ratio, output_file_name)
        
        # Run the command
        subprocess.run(command.split(sep=' '))

    #Do it with matrix of bytes, not file
    @staticmethod
    def serpentine(file):
        #A first step to convert the file into a byte matrix should be taken into account
        #This code splits the diagonals into positive and negative and adds them to a 1D array
        file_1D = []
        rows, cols = len(file), len(file[0])
        for diag in range(rows+cols-1): #Number of diagonals in mxn matrix is m + n -1
            if diag % 2 == 0:
                row = min(diag, rows - 1) #Rows -1 to not surpass limits
                col = diag - row
                while row >= 0 and col < cols:
                    file_1D.append(file[row][col])
                    row -= 1
                    col += 1      
            else:
                col = min(diag, cols - 1) #Columns -1 to not surpass limits
                row = diag - col 
                while col >= 0 and row < rows:
                    file_1D.append(file[row][col]) 
                    row += 1
                    col -= 1
        return file_1D

    @staticmethod
    def compress_hard(image_file_name):
        # Get the absolute path based on the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__)) #we mus include the absolute path to work with any computer
        abs_image_path = os.path.join(script_dir, image_file_name)

        # Check if the file exists
        if not os.path.isfile(abs_image_path):
            print(f"Error: The file '{image_file_name}' does not exist.")
            return
        
        # Prepare the ffmpeg command for compression
        # -vf format=gray declares grayscale, maskfun is used to split pixels into only white or black, threshold is 128
        # -qscale:v 31 applies the maximum jpg compression and -preset slow puts quality above execution speed
        output_file_name = os.path.join(
            script_dir,
            "".join(image_file_name.split('.')[:-1]) + '_compressed.jpg'
        )
        command = [
            'ffmpeg', '-i', abs_image_path,
            '-vf', 'format=gray,maskfun=low=128:high=128:fill=0:sum=128',
            '-qscale:v', '31', '-preset', 'slow', output_file_name, '-y'
        ]
        
        # Run the command
        subprocess.run(command)

    @staticmethod
    def run_length_encoding(bytes):
        zeros = 0
        encoded_bytes = []
        for byte in bytes:
            if byte == 0:
                zeros += 1
            else:
                if zeros > 0:
                    encoded_bytes.append(0)
                    encoded_bytes.append(zeros)

                encoded_bytes.append(byte)
                zeros = 0

        return encoded_bytes

'''
rgb = np.array([100, 54, 206])
yuv = Image.yuv_from_rgb(rgb)
rgb2 = Image.rgb_from_yuv(yuv)
print(yuv)
print(rgb2)

Image.downsize('img.png', 3)

file_matrix = np.array(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ]
    )
array_1D = Image.serpentine(file_matrix)

print(array_1D)

Image.compress_hard('img.png')

rle = [0,0,0,5,4,70,0,0,2,0,45,9,0,0,0,8]
print(rle)
print(Image.run_length_encoding(rle))

dct_test.test_dct_instance_method()
dwt_test.test_dwt_instance_method()
'''
# docker run jrottenberg/ffmpeg:4.4-alpine
# docker save jrottenberg/ffmpeg > file.tar
# docker load < file.tar

