# Run this on Laptop A
import cv2
import subprocess

rtsp_url = "rtsp://admin:admin123@192.168.128.10:554/h264/ch1/main/av_stream"

# Re-stream using ffmpeg to a local port accessible on the network
subprocess.run([
    "ffmpeg",
    "-i", rtsp_url,
    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    "rtsp://0.0.0.0:8554/cam"
])