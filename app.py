import os
import cv2
import torch
import tempfile
import requests
import base64
import mimetypes
import urllib.parse
import instaloader
from ultralytics import YOLO
import gradio as gr
import numpy as np
from PIL import Image
from pytube import YouTube
from io import BytesIO
from typing import Optional, Tuple, Union

def load_yolov8_model(model_name: str = 'yolov8s.pt') -> Optional[YOLO]:
    try:
        return YOLO(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def detect_objects(image: np.ndarray, model: YOLO) -> Optional[np.ndarray]:
    try:
        results = model(image)
        
        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            label = model.names[int(result.cls)]
            confidence = result.conf.item()

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f'{label} {confidence:.2f}'
            cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image
    except Exception as e:
        print(f"Error during object detection: {e}")
        return None

def process_image_with_yolov8(model_name: str, image: Optional[Image.Image] = None, url: Optional[str] = None) -> Optional[np.ndarray]:
    model = load_yolov8_model(model_name)
    if model is None:
        return None
    
    try:
        if url:
            response = requests.get(url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        
        if image is None:
            raise ValueError("No image provided")
        
        return detect_objects(np.array(image), model)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def download_image(url: str, save_path: str) -> None:
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image.save(save_path)
        print(f"Image saved to {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve image. Error: {e}")

def download_base64_image(base64_data: str, save_path: str) -> None:
    try:
        header, encoded = base64_data.split(',', 1)
        data = base64.b64decode(encoded)
        image = Image.open(BytesIO(data))
        image.save(save_path)
        print(f"Image saved to {save_path}")
    except Exception as e:
        print(f"Failed to decode base64 image. Error: {e}")

def download_video(url: str, save_path: str) -> None:
    try:
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path=save_path)
        print(f"Video saved to {os.path.join(save_path, stream.default_filename)}")
    except Exception as e:
        print(f"Failed to retrieve video. Error: {e}")

def download_instagram_post(url: str, save_path: str) -> None:
    loader = instaloader.Instaloader()
    post = instaloader.Post.from_shortcode(loader.context, url.split("/")[-2])
    if post.is_video:
        loader.download_post(post, target=save_path)
        print(f"Video saved to {save_path}")
    else:
        for file in post.get_sidecar_nodes():
            file_url = file.display_url
            download_image(file_url, os.path.join(save_path, f'insta_image_{file.shortcode}.jpg'))
        print(f"Images saved to {save_path}")

def download_youtube_video(youtube_url: str) -> Optional[str]:
    try:
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(file_extension='mp4', res="720p").first()
        if not stream:
            stream = yt.streams.filter(file_extension='mp4').first()
        output_path = os.path.join(tempfile.gettempdir(), "downloaded_video.mp4")
        stream.download(output_path=os.path.dirname(output_path), filename=os.path.basename(output_path))
        return output_path
    except Exception as e:
        print(f"Failed to retrieve video from YouTube. Error: {e}")
        return None

def process_video(input_video_path: str, output_video_path: str, model: YOLO) -> Optional[str]:
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_video_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detect_objects(frame, model)
        out.write(processed_frame)

    cap.release()
    out.release()

    return output_video_path

def handle_input_selection(model_name: str, choice: str, image: Optional[Image.Image], url: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    if choice == "Upload" and image is not None:
        return process_image_with_yolov8(model_name, image=image), None
    elif choice == "URL" and url:
        return process_image_with_yolov8(model_name, url=url), None
    else:
        return None, "Invalid input. Please provide an image or URL."

def process_input(model_name: str, local_video: Optional[str] = None, video_url: Optional[str] = None) -> Union[str, Optional[str]]:
    model = load_yolov8_model(model_name)
    if model is None:
        return "Failed to load YOLO model."
    
    output_path = os.path.join(tempfile.gettempdir(), "processed_video.mp4")
    
    if local_video:
        input_path = local_video
    elif video_url:
        input_path = download_youtube_video(video_url)
    else:
        return "Please provide a video file or YouTube URL."
    
    if input_path is None:
        return "Error downloading or accessing video."
    
    result = process_video(input_path, output_path, model)
    return result if result else "Error processing video."

with gr.Blocks() as demo:
    gr.Markdown("## Multimedia Object Detection Using YOLO")
    
    model_choice = gr.Radio(["yolov8s.pt", "yolov8m.pt"], label="Select Model")
    
    with gr.Tabs() as tabs:
        with gr.TabItem("Image Detection"):
            input_choice = gr.Radio(["Upload", "URL"], label="Select Input Method")
            with gr.Row():
                with gr.Column(visible=False) as upload_section:
                    input_image = gr.Image(type="pil", label="Upload Image")
                with gr.Column(visible=False) as url_section:
                    input_url = gr.Textbox(label="Image URL", placeholder="Enter URL of the image")
            
            def toggle_inputs(choice):
                return (
                    gr.update(visible=choice == "Upload"),
                    gr.update(visible=choice == "URL")
                )
            
            input_choice.change(toggle_inputs, input_choice, [upload_section, url_section])
            output_image = gr.Image(type="numpy", label="Processed Image")
            process_button_image = gr.Button("Process Image")
            process_button_image.click(handle_input_selection, [model_choice, input_choice, input_image, input_url], [output_image, gr.State()])
        
        with gr.TabItem("Video Detection"):
            local_video = gr.Video(label="Upload Local Video")
            video_url = gr.Textbox(label="YouTube Video URL")
            output_video = gr.Video(label="Processed Video")
            process_button_video = gr.Button("Process Video")
            process_button_video.click(process_input, [model_choice, local_video, video_url], output_video)

    gr.Markdown("Upload an image, provide an image URL, or provide a YouTube video URL to perform object detection using YOLOv8 model.")
    
demo.launch()
