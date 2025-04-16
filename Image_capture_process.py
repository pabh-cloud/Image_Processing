import streamlit as st
st.set_page_config(page_title="Photo & Video Processor", layout="wide")

import cv2
import numpy as np
import os
import tempfile
from datetime import datetime
from PIL import Image
import glob
import time

# Directories
CAPTURE_DIR = "captured_images"
SAVE_DIR = "processed_images"
VIDEO_DIR = "recorded_videos"
os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# App state
if 'app_state' not in st.session_state:
    st.session_state.app_state = "capture"
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'image_path' not in st.session_state:
    st.session_state.image_path = None
if 'processing_options' not in st.session_state:
    st.session_state.processing_options = {}
if 'video_mode' not in st.session_state:
    st.session_state.video_mode = "normal"  # normal, tracked
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'recording_frames' not in st.session_state:
    st.session_state.recording_frames = []
if 'tracking_box' not in st.session_state:
    st.session_state.tracking_box = None
if 'tracker' not in st.session_state:
    st.session_state.tracker = None
if 'face_tracking' not in st.session_state:
    st.session_state.face_tracking = False

# Face detection using OpenCV's Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function: Process Image (for still image processing only)
def process_image(image, options):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    processed = image.copy()

    if options.get('enable_crop', False):
        h, w = processed.shape[:2]
        crop_percent = options.get('crop_percent', 10)
        crop_px_h = int(h * crop_percent / 100)
        crop_px_w = int(w * crop_percent / 100)
        processed = processed[crop_px_h:h-crop_px_h, crop_px_w:w-crop_px_w]
    
    if options.get('apply_grayscale', False):
        processed = cv2.cvtColor(cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    
    if options.get('apply_filter', False):
        filter_color = options.get('filter_color', "#00FFFF")
        filter_intensity = options.get('filter_intensity', 0.3)
        filter_rgb = (int(filter_color[1:3], 16), int(filter_color[3:5], 16), int(filter_color[5:7], 16))
        filter_overlay = np.ones_like(processed) * filter_rgb
        processed = cv2.addWeighted(processed, 1 - filter_intensity, filter_overlay, filter_intensity, 0)

    if options.get('add_blur', False):
        blur_amount = options.get('blur_amount', 5)
        blur_val = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
        processed = cv2.GaussianBlur(processed, (blur_val, blur_val), 0)
    
    if options.get('add_edge_detection', False):
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if options.get('adjust_brightness', False):
        brightness = options.get('brightness', 0)
        contrast = options.get('contrast', 0)
        processed = cv2.convertScaleAbs(processed, alpha=(contrast + 100)/100, beta=brightness)

    return processed

# Function: Detect faces in frame
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    return faces

# Function: Initialize CSRT tracker
def init_tracker():
    return cv2.TrackerCSRT_create()

# Function: Save video from frames
def save_video_from_frames(frames, filename):
    if not frames:
        return None
    
    # Get shape from first frame
    height, width, _ = frames[0].shape
    fps = 20  # Frames per second
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_path = os.path.join(VIDEO_DIR, f"{filename}.avi")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in frames:
        out.write(frame)
    
    out.release()
    return video_path

# Function: Live Video + Capture with Face Tracking
def show_live_video():
    st.header("📹 Live Camera Feed")
    
    # Camera source selection
    camera_source = st.radio("Camera Source", ["Laptop/USB Camera", "Smartphone (IP Webcam)"])
    
    # Camera controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        run = st.checkbox("Start Camera", value=False, key="camera_start")
    
    with col2:
        video_mode = st.selectbox("Display Mode", 
                              ["Normal", "Face Tracking"], 
                              index=0,
                              key="display_mode")
        st.session_state.video_mode = video_mode.lower()
    
    with col3:
        if camera_source == "Laptop/USB Camera":
            camera_id = st.selectbox("Camera", options=[0, 1, 2, 3], index=0, 
                            help="Select camera device ID (0 is usually the default webcam)",
                            key="camera_id")
            camera_source_value = camera_id
        else:
            ip_url = st.text_input("IP Webcam URL", "http://192.168.81.70:8080/video", 
                         help="Enter the URL of your IP Webcam stream",
                         key="ip_url")
            camera_source_value = ip_url

    # Video display area
    FRAME_WINDOW = st.empty()
    
    # Status area
    status_area = st.empty()
    
    # Control buttons
    button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
    capture_button = button_col1.button("📸 Capture Frame", key="capture_button")
    record_button = button_col2.button("⏺️ Start/Stop Recording", key="record_button")
    switch_button = button_col3.button("🔄 Switch to Processing", key="switch_button")
    
    # Initialize state variables
    if run:
        # Store in session state to persist between reruns
        if 'recording' not in st.session_state:
            st.session_state.recording = False
        if 'recording_frames' not in st.session_state:
            st.session_state.recording_frames = []
        if 'start_time' not in st.session_state:
            st.session_state.start_time = None
        if 'face_tracking' not in st.session_state:
            st.session_state.face_tracking = False
        if 'tracking_box' not in st.session_state:
            st.session_state.tracking_box = None
        if 'tracker' not in st.session_state:
            st.session_state.tracker = None
            
        # Initialize camera
        camera = cv2.VideoCapture(camera_source_value)
        if not camera.isOpened():
            st.error(f"Could not open camera source: {camera_source_value}")
            return
        
        # Set better resolution if possible
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Process button inputs outside the camera loop
        if switch_button:
            st.session_state.app_state = "process"
            camera.release()
            st.rerun()
            
        if record_button:
            # Toggle recording state
            st.session_state.recording = not st.session_state.recording
            if st.session_state.recording:
                st.session_state.recording_frames = []
                st.session_state.start_time = time.time()
                status_area.info("Started recording...")
            else:
                # Save video if we have frames
                if st.session_state.recording_frames:
                    video_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    video_path = save_video_from_frames(st.session_state.recording_frames, video_filename)
                    status_area.success(f"Video saved as: {video_path}")
                    st.session_state.recording_frames = []
                    st.session_state.start_time = None
        
        try:
            # Single frame capture for the first run of the loop
            ret, frame = camera.read()
            if not ret:
                status_area.warning("Failed to read frame.")
                camera.release()
                return
                
            # Main camera loop using a placeholder to avoid rerunning the app
            frame_placeholder = st.empty()
            
            # Process and display the first frame
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Face detection for the first frame if in tracking mode
            if st.session_state.video_mode == "face tracking":
                faces = detect_faces(frame)
                # Draw rectangle around detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                # Initialize tracker with the first face if found and not already tracking
                if len(faces) > 0 and not st.session_state.face_tracking:
                    x, y, w, h = faces[0]
                    st.session_state.tracker = init_tracker()
                    st.session_state.tracking_box = (x, y, w, h)
                    ok = st.session_state.tracker.init(frame, st.session_state.tracking_box)
                    st.session_state.face_tracking = ok
            
            FRAME_WINDOW.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Capture button handling
            if capture_button:
                capture_filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                capture_path = os.path.join(CAPTURE_DIR, capture_filename)
                cv2.imwrite(capture_path, frame)  # Save original frame
                st.session_state.captured_image = capture_path
                st.session_state.image_path = capture_path
                st.session_state.app_state = "process"
                camera.release()
                st.rerun()
                
            # Stop button
            stop_button = st.button("Stop Camera", key="stop_camera")
            if stop_button:
                camera.release()
                st.rerun()
                return
                
            # Manual frame processing loop
            frame_count = 0
            while run and not stop_button and frame_count < 1000:  # Limit frames to prevent browser crashes
                ret, frame = camera.read()
                if not ret:
                    break
                    
                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()
                
                # Face tracking
                if st.session_state.video_mode == "face tracking":
                    if st.session_state.face_tracking:
                        # Update tracker
                        success, box = st.session_state.tracker.update(frame)
                        
                        if success:
                            # Draw tracking box
                            x, y, w, h = [int(v) for v in box]
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(display_frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            # Tracking failed, try to detect faces again
                            st.session_state.face_tracking = False
                            faces = detect_faces(frame)
                            
                            if len(faces) > 0:
                                # Re-initialize tracker with the first face
                                x, y, w, h = faces[0]
                                st.session_state.tracker = init_tracker()
                                st.session_state.tracking_box = (x, y, w, h)
                                ok = st.session_state.tracker.init(frame, st.session_state.tracking_box)
                                st.session_state.face_tracking = ok
                                
                                # Draw rectangle around detected face
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(display_frame, "Re-detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            else:
                                cv2.putText(display_frame, "Face lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        # Try to detect faces
                        faces = detect_faces(frame)
                        
                        if len(faces) > 0:
                            # Initialize tracker with the first face
                            x, y, w, h = faces[0]
                            st.session_state.tracker = init_tracker()
                            st.session_state.tracking_box = (x, y, w, h)
                            ok = st.session_state.tracker.init(frame, st.session_state.tracking_box)
                            st.session_state.face_tracking = ok
                            
                            # Draw rectangle around detected face
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(display_frame, "Face detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            cv2.putText(display_frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display frame
                FRAME_WINDOW.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                
                # Update recording if active
                if st.session_state.recording:
                    st.session_state.recording_frames.append(display_frame.copy())  # Save the displayed frame (with tracking boxes)
                    
                    # Update recording status
                    elapsed = time.time() - st.session_state.start_time
                    status_area.info(f"Recording... {elapsed:.1f}s | {len(st.session_state.recording_frames)} frames")
                
                frame_count += 1
                time.sleep(0.03)  # ~30 FPS
                
            # Auto-save video if recording is still active
            if st.session_state.recording and st.session_state.recording_frames:
                video_filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                video_path = save_video_from_frames(st.session_state.recording_frames, video_filename)
                status_area.success(f"Video saved as: {video_path}")
                st.session_state.recording_frames = []
                st.session_state.recording = False
                
        except Exception as e:
            st.error(f"Error in video processing: {e}")
        finally:
            camera.release()

# Sidebar UI
st.sidebar.title("Photo & Video Processor")

if st.session_state.app_state == "capture":
    st.sidebar.header("Camera Options")
    
    if st.sidebar.button("Skip to Image Processing", key="skip_to_processing"):
        st.session_state.app_state = "process"
    
    # Show live video interface
    show_live_video()

else:  # process mode
    st.sidebar.header("Image Processing")
    st.sidebar.subheader("Image Source")
    image_source = st.sidebar.radio("Select Image Source", ["Captured Image", "Upload Image", "Existing Images"], key="image_source")

    if image_source == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], key="file_uploader")
        if uploaded_file is not None:
            st.session_state.image_path = uploaded_file
    elif image_source == "Existing Images":
        captured_images = glob.glob(f"{CAPTURE_DIR}/*.jpg") + glob.glob(f"{CAPTURE_DIR}/*.jpeg") + glob.glob(f"{CAPTURE_DIR}/*.png")
        selected_image = st.sidebar.selectbox("Select existing image", ["None"] + captured_images, key="select_existing")
        if selected_image != "None":
            st.session_state.image_path = selected_image
    else:
        if st.session_state.captured_image:
            st.session_state.image_path = st.session_state.captured_image
        else:
            st.sidebar.warning("No image has been captured yet.")

    # Processing options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Processing Options")
    options = {}
    options['apply_grayscale'] = st.sidebar.checkbox("Convert to Grayscale", key="proc_grayscale")
    options['apply_filter'] = st.sidebar.checkbox("Apply Color Filter", key="proc_filter")
    if options['apply_filter']:
        options['filter_color'] = st.sidebar.color_picker("Filter Color", "#00FFFF", key="proc_filter_color")
        options['filter_intensity'] = st.sidebar.slider("Filter Intensity", 0.0, 1.0, 0.3, 0.1, key="proc_filter_intensity")
    options['enable_crop'] = st.sidebar.checkbox("Enable Cropping", key="proc_crop")
    if options['enable_crop']:
        options['crop_percent'] = st.sidebar.slider("Crop Percentage", 0, 50, 10, 5, key="proc_crop_percent")
    options['add_blur'] = st.sidebar.checkbox("Add Blur Effect", key="proc_blur")
    if options['add_blur']:
        options['blur_amount'] = st.sidebar.slider("Blur Amount", 1, 21, 5, 2, key="proc_blur_amount")
    options['add_edge_detection'] = st.sidebar.checkbox("Edge Detection", key="proc_edge")
    options['adjust_brightness'] = st.sidebar.checkbox("Adjust Brightness/Contrast", key="proc_brightness")
    if options['adjust_brightness']:
        options['brightness'] = st.sidebar.slider("Brightness", -100, 100, 0, 5, key="proc_brightness_val")
        options['contrast'] = st.sidebar.slider("Contrast", -100, 100, 0, 5, key="proc_contrast")

    st.session_state.processing_options = options

    if st.sidebar.button("Back to Camera Mode", key="back_to_camera"):
        st.session_state.app_state = "capture"
        st.rerun()

# Main
def main():
    if st.session_state.app_state == "process":
        st.header("📷 Image Processing")
        
        image = None
        path = st.session_state.image_path
        if isinstance(path, str) and os.path.exists(path):
            image = cv2.imread(path)
        elif path is not None:
            try:
                image = Image.open(path)
                image = np.array(image)
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            except Exception as e:
                st.error(f"Error loading image: {e}")

        if image is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

            processed = process_image(image, st.session_state.processing_options)

            with col2:
                st.subheader("Processed Image")
                st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_column_width=True)

            st.subheader("Save Options")
            filename = st.text_input("Filename", value=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}", key="save_filename")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Processed Image", key="save_image"):
                    save_path = os.path.join(SAVE_DIR, f"{filename}.jpg")
                    cv2.imwrite(save_path, processed)
                    st.success(f"Image saved to {save_path}")
                    with open(save_path, "rb") as file:
                        st.download_button("Download Processed Image", file, file_name=f"{filename}.jpg", mime="image/jpeg", key="download_image")
            
            with col2:
                if st.button("🔄 Back to Camera", key="back_to_camera_from_process"):
                    st.session_state.app_state = "capture"
                    st.rerun()
        else:
            st.info("Please capture or select an image to proceed.")

# Add instructions
def show_instructions():
    with st.expander("📖 How to Use This App"):
        st.markdown("""
        ### Camera Options
        - **Laptop/USB Camera**: Uses your device's built-in webcam or USB camera
        - **Smartphone Camera**: Connect your phone's camera using an IP Webcam app
          - Install "IP Webcam" app on your Android phone
          - Start the server in the app
          - Enter the URL (usually http://[your-phone-ip]:8080/video)
        
        ### Features
        - **Face Tracking**: Automatically detects and tracks faces using CSRT tracking
        - **Capture**: Save still frames for further editing
        - **Record**: Record videos with face tracking boxes
        - **Image Processing**: Edit captured or uploaded images with various filters and effects
        
        ### Tips
        - For best face tracking, ensure good lighting conditions
        - If using smartphone camera, ensure both devices are on the same network
        - For image processing, try different combinations of effects for creative results
        """)

# Run app
if __name__ == "__main__":
    show_instructions()
    main()