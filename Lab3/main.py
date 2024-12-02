import numpy as np
from scipy import linalg
import subprocess
import os
import dct_test
import dwt_test
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
from typing import List
from pathlib import Path

app = FastAPI()

# Directory for storing uploads and outputs
UPLOAD_DIR = Path("/app/uploads")
OUTPUT_DIR = Path("/app/output")

# Ensure the directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Serve static files from the /app/output directory
app.mount("/output", StaticFiles(directory="/app/output"), name="output")

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

class CompressModel(BaseModel):
    image_file_name: str

class RLEModel(BaseModel):
    bytes: List[int]

# Serve the central navigation page
@app.get("/", response_class=HTMLResponse)
async def get_navigation():
    with open("main.html") as f:
        return HTMLResponse(content=f.read())

# Serve the static HTML file
@app.get("/yuv_from_rgb", response_class=HTMLResponse)
async def get_index():
    with open("yuv_from_rgb.html") as f:
        return HTMLResponse(content=f.read())
    
# Serve the static HTML file
@app.get("/rgb_from_yuv", response_class=HTMLResponse)
async def get_index():
    with open("rgb_from_yuv.html") as f:
        return HTMLResponse(content=f.read())

# Serve the static HTML file
@app.get("/downsize", response_class=HTMLResponse)
async def get_index():
    with open("downsize.html") as f:
        return HTMLResponse(content=f.read())
    
# API endpoint for serving the downsized image
@app.get("/output/{filename}")
async def get_downsized_image(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
    
# Serve the static HTML file
@app.get("/serpentine", response_class=HTMLResponse)
async def get_index():
    with open("serpentine.html") as f:
        return HTMLResponse(content=f.read())

# Serve the static HTML file
@app.get("/compress_hard", response_class=HTMLResponse)
async def get_index():
    with open("compress_hard.html") as f:
        return HTMLResponse(content=f.read())

# Serve the static HTML file
@app.get("/run_length_encoding", response_class=HTMLResponse)
async def get_index():
    with open("run_length_encoding.html") as f:
        return HTMLResponse(content=f.read())
    
# Serve the static HTML file
@app.get("/resize_video", response_class=HTMLResponse)
async def get_index():
    with open("video_resize.html") as f:
        return HTMLResponse(content=f.read())
    
# Serve the static HTML file
@app.get("/chroma_subsampling", response_class=HTMLResponse)
async def get_index():
    with open("chroma_subsampling.html") as f:
        return HTMLResponse(content=f.read())

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
async def downsize_image(image: UploadFile = File(...), ratio: int = 2):
    try:
        # Save the uploaded image to the server
        input_image_path = UPLOAD_DIR / image.filename
        with open(input_image_path, "wb") as buffer:
            buffer.write(await image.read())

        # Call the downsizing function
        downsized_image_path = Image.downsize(input_image_path, ratio)
        
        # Return the downsized image URL
        return {
            "success": True,
            "downsized_image_url": f"/output/{downsized_image_path.name}"
        }
    except Exception as e:
        raise HTTPException(status_code = 500, detil = str(e))

@app.post("/serpentine")
def serpentine(serpentine_data: SerpentineModel):
    try:
        result = Image.serpentine(serpentine_data.file)
        return {"serpentine": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compress_hard")
async def compress_hard_endpoint(image: UploadFile = File(...)):
    try:
        # Save the uploaded image to the server
        input_image_path = UPLOAD_DIR / image.filename
        with open(input_image_path, "wb") as buffer:
            buffer.write(await image.read())

        # Call the compress_hard method to compress the image
        compressed_image_path = Image.compress_hard(input_image_path)

        # Return a URL to the compressed image
        compressed_image_url = f"/output/{compressed_image_path.name}"
        return {
            "success": True,
            "compressed_image_url": compressed_image_url
        }

    except HTTPException as e:
        # If there was an error (e.g., file not found or ffmpeg error), it will be caught here
        raise e
    except Exception as e:
        # Catch any other errors and raise as HTTPException
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")

@app.post("/run_length_encoding")
def run_length_encoding(rle_data: RLEModel):
    try:
        encoded_result = Image.run_length_encoding(rle_data)
        return {"success": True, "encoded_result": encoded_result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error encoding values: {str(e)}")

@app.post("/resize_video")
async def resize_video_endpoint(video: UploadFile = File(...), resolution: str = "426:240"):
    try:
        # Save the uploaded image to the server
        input_video_path = UPLOAD_DIR / video.filename
        with open(input_video_path, "wb") as buffer:
            buffer.write(await video.read())

        # Call the modify_resolution method 
        resized_video_path = Image.video_resize(input_video_path, resolution)

        # Return a URL to the compressed image
        resized_video_url = f"/output/{resized_video_path.name}"
        print(f"Recieved resolution: {resolution}")
        return {
            "success": True,
            "resized_video_url": resized_video_url
        }

    except HTTPException as e:
        # If there was an error (e.g., file not found or ffmpeg error), it will be caught here
        raise e
    except Exception as e:
        # Catch any other errors and raise as HTTPException
        raise HTTPException(status_code=500, detail=f"Resize failed: {str(e)}")

@app.post("/chroma_subsampling")
async def chroma_subsampling(video: UploadFile = File(...), subsampling_factor: str = "yuv420p"):
    try:
        # Log the received subsampling factor and video file name
        print(f"Received subsampling factor: {subsampling_factor}")
        
        # Save the uploaded video to the server
        input_video_path = UPLOAD_DIR / video.filename
        with open(input_video_path, "wb") as buffer:
            buffer.write(await video.read())
        
        # Log the saved input file path
        print(f"Saved video to: {input_video_path}")

        # Call the chroma_subsampling method with the provided subsampling factor
        chroma_subsampled_video_path = Image.chroma_subsampling(input_video_path, subsampling_factor)

        # Log the result video path
        print(f"Subsampled video saved to: {chroma_subsampled_video_path}")

        # Return a URL to the subsampled video
        chroma_subsampled_video_url = f"/output/{chroma_subsampled_video_path.name}"
        return {
            "success": True,
            "chroma_subsampled_video_url": chroma_subsampled_video_url
        }

    except HTTPException as e:
        # If there was an error (e.g., file not found or FFmpeg error), it will be caught here
        print(f"HTTPException: {e.detail}")
        raise e
    except Exception as e:
        # Catch any other errors and raise as HTTPException
        print(f"Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chroma Subsampling failed: {str(e)}")

class Image:
    yuv_from_rgb_mat = np.array(
        [
            [0.257, 0.504, 0.098],
            [-0.148, -0.291, 0.439],
            [0.439, -0.368, -0.071],
        ]
    )

    rgb_from_yuv_mat = np.array(
        [
            [1.164, 0, 1.596],     # R
            [1.164, -0.392, -0.813],  # G
            [1.164, 2.017, 0],     # B
        ]
    )

    # Offset arrays for the YUV and RGB conversions
    yuv_offset = np.array([16, 128, 128])

    @staticmethod
    def yuv_from_rgb(rgb):     
        yuv = (rgb @ Image.yuv_from_rgb_mat.T) + Image.yuv_offset
        return yuv.astype(int)

    @staticmethod
    def rgb_from_yuv(yuv):
        rgb = (yuv - Image.yuv_offset) @ Image.rgb_from_yuv_mat.T
        rgb = np.clip(rgb, 0, 255)
        return rgb.astype(int)
    
    @staticmethod
    def downsize(input_image_path, ratio = 2):
        """Downsize the image using ffmpeg"""
        output_image_path = OUTPUT_DIR / f"{input_image_path.stem}_small.jpg"

        # Build the ffmpeg command
        command = f'ffmpeg -i "{input_image_path}" -vf scale=iw/{ratio}:ih/{ratio} "{output_image_path}" -y'
        
        # Run the command to downsize the image
        subprocess.run(command, shell=True)

        # Check if the output image was created successfully
        if not output_image_path.exists():
            raise HTTPException(status_code=500, detail="Downsized image creation failed.")

        return output_image_path

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
    def compress_hard(input_image_path):
        try:
            output_image_path = OUTPUT_DIR / f"{input_image_path.stem}_compressed.jpg"
            # ffmpeg command to compress the image
            command = [
                'ffmpeg', '-i', input_image_path,
                '-vf', 'format=gray,maskfun=low=128:high=128:fill=0:sum=128',  # Compression filters
                '-qscale:v', '31',  # Compression quality
                '-preset', 'slow',  # Compression speed (slow for quality)
                output_image_path,
                '-y'  # Overwrite output file if it exists
            ]

            # Run the command
            result = subprocess.run(command, capture_output=True, text=True)

            # If there was an error during compression
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")

            return output_image_path  # Return the path to the compressed file

        except FileNotFoundError as fnf_error:
            raise HTTPException(status_code=404, detail=str(fnf_error))

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")


    @staticmethod
    def run_length_encoding(rle_data):
        input_bytes = rle_data.bytes
        zeros = 0
        encoded_bytes = []
        for byte in input_bytes:
            if byte == 0:
                zeros += 1
            else:
                if zeros > 0:
                    encoded_bytes.append(0)
                    encoded_bytes.append(zeros)

                encoded_bytes.append(byte)
                zeros = 0
            # Check for any trailing zeros after the loop ends
        if zeros > 0:
            encoded_bytes.append(0)
            encoded_bytes.append(zeros)

        return encoded_bytes

    @staticmethod
    def video_resize(input_video_path, new_resolution):
        try:
            # Construct the output file path
            output_video_path = OUTPUT_DIR / f"{input_video_path.stem}_{new_resolution.replace(':', 'x')}.mp4"
            
            # FFmpeg command to change the resolution
            command = [
                'ffmpeg', '-i', str(input_video_path),  # Input video
                '-vf', f'scale={new_resolution}',      # Video filter to change resolution
                '-c:v', 'libx264',                     # Use H.264 codec for better compatibility
                '-preset', 'slow',                     # Compression speed (slow for quality)
                '-crf', '23',                          # Constant Rate Factor (quality, lower = better)
                str(output_video_path),                # Output video
                '-y'                                   # Overwrite output file if it exists
            ]

            # Run the FFmpeg command
            result = subprocess.run(command, capture_output=True, text=True)

            # If there was an error during execution
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")

            return output_video_path  # Return the path to the resized video file

        except FileNotFoundError as fnf_error:
            raise HTTPException(status_code=404, detail=str(fnf_error))

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Resolution change failed: {str(e)}")
        
    @staticmethod
    def chroma_subsampling(input_video_path, subsampling_factor):
        try:
            # Construct the output file path
            output_video_path = OUTPUT_DIR / f"{input_video_path.stem}_{subsampling_factor}.mp4"
            
            # FFmpeg command to change the resolution
            command = [
                'ffmpeg', '-i', str(input_video_path),  # Input video
                '-c:v', 'libx264',                      # Use H.264 codec for better compatibility
                '-preset', 'slow',                      # Compression speed (slow for better quality)
                '-crf', '23',                           # Constant Rate Factor (quality, lower = better)
                '-pix_fmt', str(subsampling_factor),    # Apply chroma subsampling 
                str(output_video_path),                 # Output video path
                '-y'                                    # Overwrite output file if it exists
            ]

            # Run the FFmpeg command
            result = subprocess.run(command, capture_output=True, text=True)

            # If there was an error during execution
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")

            return output_video_path  # Return the path to the resized video file

        except FileNotFoundError as fnf_error:
            raise HTTPException(status_code=404, detail=str(fnf_error))

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Resolution change failed: {str(e)}")