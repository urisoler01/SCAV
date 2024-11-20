import json

import numpy as np
from scipy import linalg
import subprocess
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()


# Pydantic models for request bodies
class RGBModel(BaseModel):
    rgb: List[int]

class YUVModel(BaseModel):
    yuv: List[int]

class DownsizeModel(BaseModel):
    image_file_name: str
    ratio: int = 2

class SerpentineModel(BaseModel):
    file: List[List[int]]

class FileModel(BaseModel):
    file_name: str

class RLEModel(BaseModel):
    bytes: List[int]

@app.post("/yuv_from_rgb")
def yuv_from_rgb(rgb_data: RGBModel):
    if len(rgb_data.rgb) != 3:
        raise HTTPException(status_code=400, detail="RGB must have exactly 3 values")
    rgb = np.array(rgb_data.rgb)
    yuv = Image.yuv_from_rgb(rgb)
    return {"yuv": yuv.tolist()}

@app.post("/rgb_from_yuv")
def rgb_from_yuv(yuv_data: YUVModel):
    if len(yuv_data.yuv) != 3:
        raise HTTPException(status_code=400, detail="YUV must have exactly 3 values")
    yuv = np.array(yuv_data.yuv)
    rgb = Image.rgb_from_yuv(yuv)
    return {"rgb": rgb.tolist()}

@app.post("/downsize")
def downsize(downsize_data: DownsizeModel):
    try:
        Image.downsize(downsize_data.image_file_name, downsize_data.ratio)
        return {"message": f"Image {downsize_data.image_file_name} downsized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/serpentine")
def serpentine(serpentine_data: SerpentineModel):
    try:
        result = Image.serpentine(serpentine_data.file)
        return {"serpentine": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compress_hard")
def compress_hard(compress_data: FileModel):
    try:
        Image.compress_hard(compress_data.file_name)
        return {"message": f"Image {compress_data.file_name} compressed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_length_encoding")
def run_length_encoding(rle_data: RLEModel):
    try:
        result = Image.run_length_encoding(rle_data.bytes)
        return {"encoded": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_info")
def get_info(file_data: FileModel):
    try:
        info = Image.get_info(file_data.file_name)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bbb_endpoint")
def bbb_endpoint(file_data: FileModel):
    try:
        Image.bbb_endpoint(file_data.file_name)
        return {"message": f"BBB endpoint processing completed for {file_data.file_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tracknumber")
def tracknumber(file_data: FileModel):
    try:
        tracks = Image.tracknumber(file_data.file_name)
        return {"tracks": tracks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/show_motionvectors")
def show_motionvectors(file_data: FileModel):
    try:
        Image.show_motionvectors(file_data.file_name)
        return {"message": f"Motion vectors visualization created for {file_data.file_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/yuv_histogram")
def yuv_histogram(file_data: FileModel):
    try:
        Image.yuv_histogram(file_data.file_name)
        return {"message": f"YUV histogram video created for {file_data.file_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class Image:
    yuv_from_rgb_mat = (1/255) * np.array(
        [
            [66.5, 129, 25],
            [-38, -74, 112],
            [112, -94, -18],
        ]
    )

    '''rgb_from_yuv_mat = np.array(
        [
            [1, 0, 1.13983],
            [1, -0.39465, -0.58060],
            [1, 2.03211, 0]
        ]
    )'''

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
            ".".join(image_file_name.split('.')[:-1]) + '_small.jpg'
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
            ".".join(image_file_name.split('.')[:-1]) + '_compressed.jpg'
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

    @staticmethod
    def get_info(filename):
        # Get the absolute path based on the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__)) #we must include the absolute path to work with any computer
        abs_path = os.path.join(script_dir, filename)

        # Check if the file exists
        if not os.path.isfile(abs_path):
            print(f"Error: The file '{filename}' does not exist.")
            return


        command = ["ffprobe", "-v", "quiet", "-show_format", "-show_streams", "-print_format", "json", abs_path]
        info = subprocess.check_output(command).decode("utf-8")
        return info

    @staticmethod
    def show_motionvectors(filename):
        # Get the absolute path based on the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__)) #we must include the absolute path to work with any computer
        abs_path = os.path.join(script_dir, filename)

        # Check if the file exists
        if not os.path.isfile(abs_path):
            print(f"Error: The file '{filename}' does not exist.")
            return

        out = ".".join(abs_path.split(".")[:-1]) + "_motionvectors.mp4"

        command = ["ffmpeg", "-flags2", "+export_mvs", "-i", abs_path, "-vf", "codecview=mv=pf+bf+bb", out, "-y"]

        subprocess.check_output(command)

    @staticmethod
    def bbb_endpoint(filename):
        # Get the absolute path based on the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__)) #we must include the absolute path to work with any computer
        abs_path = os.path.join(script_dir, filename)

        # Check if the file exists
        if not os.path.isfile(abs_path):
            print(f"Error: The file '{filename}' does not exist.")
            return

        out_path = ".".join(abs_path.split(".")[:-1])
        first_out = ".".join(abs_path.split(".")[:-1]) + "1.mp4"

        command = ["ffmpeg", "-i", abs_path, "-acodec", "copy", "-f", "segment", "-segment_time", "20", "-vcodec", "copy",
                   "-reset_timestamps", "1", out_path + "%d.mp4", "-y"]
        subprocess.run(command)

        command = ["ffmpeg", "-i", first_out, "-vn", "-acodec", "aac", out_path + ".aac", "-y"]
        subprocess.run(command)

        command = ["ffmpeg", "-i", first_out, "-vn", "-acodec", "mp3", "-qscale:a", "7", out_path + ".mp3", "-y"]
        subprocess.run(command)

        command = ["ffmpeg", "-i", first_out, "-vn", "-acodec", "ac3", out_path + ".ac3", "-y"]
        subprocess.run(command)

        command = ["ffmpeg", "-i", first_out, "-i", out_path + ".aac", "-i",
                   out_path + ".mp3", "-i", out_path + ".ac3", "-c", "copy",
                    "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0", "-map", "3:a:0",
                   out_path + "_export.mp4", "-y"]
        subprocess.run(command)

    @staticmethod
    def tracknumber(filename):
        info = Image.get_info(filename)
        return len(json.loads(info)["streams"])

    @staticmethod
    def yuv_histogram(filename):
        # Get the absolute path based on the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__)) #we must include the absolute path to work with any computer
        abs_path = os.path.join(script_dir, filename)

        # Check if the file exists
        if not os.path.isfile(abs_path):
            print(f"Error: The file '{filename}' does not exist.")
            return

        out = ".".join(abs_path.split(".")[:-1]) + "_yuvhist.mp4"

        command = ["ffmpeg", "-i", abs_path, "-vf", "split=2[a][b],[b]histogram,format=yuva444p[hh],[a][hh]overlay", out, "-y"]

        subprocess.check_output(command)


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

print(Image.get_info('bbb_sunflower_1080p_30fps_normal.mp4'))
Image.bbb_endpoint('bbb_sunflower_1080p_30fps_normal.mp4')
print(Image.tracknumber('bbb_sunflower_1080p_30fps_normal_export.mp4'))
Image.show_motionvectors('bbb_sunflower_1080p_30fps_normal.mp4')
Image.yuv_histogram('bbb_sunflower_1080p_30fps_normal.mp4')
'''
# docker run jrottenberg/ffmpeg:4.4-alpine
# docker save jrottenberg/ffmpeg > file.tar
# docker load < file.tar

