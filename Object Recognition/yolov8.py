from ultralytics import YOLO
from PIL import Image

# model loading
model = YOLO('yolov8n.pt')

# object recognition in image
im1 = Image.open("Elon_Musk.jpg")
im1_result = model.predict(source=im1, save=True)

# object recognition in live cam
#cam_result = model.predict(source="0", show=True)

# object recognition in video
video_result = model.predict(source="video.mp4", show=True)