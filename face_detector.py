import cv2
import streamlit as st
import numpy as np
from numpy import ndarray
from io import BytesIO


class ImageUploadHelper(object):

    def __init__(self):
        self.fileTypes = ['jpg', 'png']

    def run(self) -> ndarray:
        """
            Upload File on Streamlit
            Return Image : ndarray
        """
        file = st.file_uploader("Upload File", type=self.fileTypes, accept_multiple_files=False)
        show_file = st.empty()

        if not file:
            show_file.info(f"Please upload a file: {','.join(self.fileTypes)} ")

        if isinstance(file, BytesIO):
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.int8)
            img = cv2.imdecode(file_bytes, 1)
            return img
        else:
            return None

def main():
    st.title("""
        A Face Detector Project 
    """)

    st.header("Detect Face on the image uploaded")

    # STEP 1: Load pre-trained data on face frontal from opencv
    trained_data_face = cv2.CascadeClassifier('./data-training/haarcascade_frontalface_default.xml')

    # STEP 2: Choose an image to detect face
    file = ImageUploadHelper().run()
    image =  file if file is not None else cv2.imread('images/img.jpg')

    # STEP 3: We don't need colors on image, must converted to grayscale
    grayscaled_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # STEP 4 : Detect Face and draw rectangle
    face_coordinates = trained_data_face.detectMultiScale(grayscaled_img)

    # Draw rectangle around faces
    for faces in face_coordinates:
        [x, y, w, h] = faces
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(image, caption="Image", channels='BGR')

if __name__ == '__main__':
    main()