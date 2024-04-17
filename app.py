import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import os
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import datetime

# Page configuration
st.set_page_config(
    page_title='Gate Pass System',
    page_icon='🙋‍♂️',
)

# Define paths to cascade classifier files
CASCADE_DIR = os.path.dirname(__file__)
FACE_CASCADE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_frontalface_default.xml')
EYE_CASCADE_PATH = os.path.join(CASCADE_DIR, 'haarcascade_eye.xml')

# Load cascade classifiers
faceCascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
eyesCascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

# Initialize session state if not present
if "records" not in st.session_state:
    st.session_state.records = []

# Define a function to detect faces and count them in an image
def detect_faces(image):
    detect_img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(detect_img, cv2.COLOR_RGB2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(detect_img, (x, y), (x + w, y + h), (255, 255, 0), 2)
    return detect_img, len(faces)

# Define a function to detect eyes and count them in an image
def detect_eyes(image):
    detect_img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(detect_img, cv2.COLOR_RGB2GRAY)
    eyes = eyesCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in eyes:
        cv2.rectangle(detect_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return detect_img, len(eyes)

# Define a function to detect faces and count them in a webcam frame
def detect_faces_webcam(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 255, 0), 2)
    # Display the number of faces detected with current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame_rgb, f'Faces Detected: {len(faces)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_rgb, f'Date and Time: {current_time}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame_rgb, len(faces)

# Define the video transformer class
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Initialize the superclass
        super().__init__()
        # Initialize attributes for frame and face count
        self.frame_with_faces = None
        self.face_count = 0

    def transform(self, frame):
        frame = np.array(frame.to_image())
        frame_with_faces, face_count = detect_faces_webcam(frame)
        
        # Update the attributes with the latest frame and face count
        self.frame_with_faces = frame_with_faces
        self.face_count = face_count
        
        return frame_with_faces

def main():
    st.title("Face Recognition based Gate Pass System for UEMJ Main Gate")
    st.write("Built with Streamlit and OpenCV")

    activities = ["Image Detection", "Webcam", "About"]
    choice = st.sidebar.selectbox("Select Activities", activities)

    if choice == "Image Detection":
        st.subheader("Face and Eye Detection from Image")
        img_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
        if img_file is not None:
            image = Image.open(img_file)
            st.image(image, caption='Original Image', use_column_width=True)

            enhance_type = st.sidebar.radio("Enhance type", ["Original", "Gray-scale", "Contrast", "Brightness", "Blurring"])
            if enhance_type == "Gray-scale":
                gray_image = image.convert('L')
                st.image(gray_image, caption='Gray-scale Image', use_column_width=True)

            if enhance_type == "Contrast":
                c_make = st.sidebar.slider("Contrast", 0.5, 3.5)
                enhancer = ImageEnhance.Contrast(image)
                enhanced_image = enhancer.enhance(c_make)
                st.image(enhanced_image, caption='Contrast Enhanced Image', use_column_width=True)

            if enhance_type == "Brightness":
                b_make = st.sidebar.slider("Brightness", 0.5, 3.5)
                enhancer = ImageEnhance.Brightness(image)
                enhanced_image = enhancer.enhance(b_make)
                st.image(enhanced_image, caption='Brightness Enhanced Image', use_column_width=True)

            if enhance_type == "Blurring":
                br_make = st.sidebar.slider("Blurring", 0.5, 3.5)
                blurred_image = image.filter(ImageFilter.GaussianBlur(radius=br_make))
                st.image(blurred_image, caption='Blurred Image', use_column_width=True)

            feature_choice = st.sidebar.radio("Find Feature", ["Faces", "Eyes"])
            if st.button("Process"):
                if feature_choice == "Faces":
                    result_img, result_count = detect_faces(image)
                    st.image(result_img, caption=f'Faces Detected: {result_count}', use_column_width=True)
                elif feature_choice == "Eyes":
                    result_img, result_count = detect_eyes(image)
                    st.image(result_img, caption=f'Eyes Detected: {result_count}', use_column_width=True)

    elif choice == "Webcam":
        st.subheader("Real-Time Face Detection")
        
        # Initialize the webrtc context
        webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
        
        # Button to record the data
        if st.button("RECORD"):
            # Get the VideoTransformer object
            video_transformer = webrtc_ctx.video_transformer
            
            # If the transformer is available, use its data
            if video_transformer:
                frame, face_count = video_transformer.frame_with_faces, video_transformer.face_count
                
                # Get current date and time
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Record the data in session state
                st.session_state.records.append({"Time": current_time, "Faces Detected": face_count})
                
                # Display a success message
                st.success(f"Data recorded at {current_time} with {face_count} face(s) detected.")
        
        # Display the recorded data in a table
        if st.session_state.records:
            st.dataframe(st.session_state.records)

    elif choice == "About":
        st.write("This Application is Developed By Alok and Shashank")

if __name__ == '__main__':
    main()
