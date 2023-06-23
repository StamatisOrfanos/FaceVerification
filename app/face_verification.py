# Import Kivy decencies 
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import libraries
import os, cv2, embedding_layer
import tensorflow as tf
import numpy as np
from L1Dist import L1Dist
from siamese import Siamese

# Load Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create the application class
class VerificationApp(App):

    def build(self):
        self.web_cam = Image(size_hint=(1,.7))
        self.button = Button(text='Verify', on_press=self.verify, size_hint=(1,.1))
        self.label = Label(text='Verification Loading', size_hint=(1,.2))

        # Define the app layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.label)

        # Load the model
        self.model = tf.keras.models.load_model('siameseModel.h5', custom_objects={'siamese': Siamese, 'L1Dist': L1Dist})

        # Implement the video capture
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0) # Run update 33 times per second

        return layout
    

    # Keep the video capture alive
    def update(self, *args):
        
        # Read the face in real time and save to the input_image folder
        ret, frame = self.capture.read()
        buffer = cv2.flip(frame, 0).tostring()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = image_texture


    # Image pre-processing
    def pre_process(self, image_path):
        # Load image
        byte_img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(byte_img)

        # Normalize in [0,1] and resize to 100x100x3 for the model
        img = tf.image.resize(img, (100,100))
        img = img / 255.0

        return img
    

    # Verification function to return if the images are to be verified
    def verify(self, *args):

        # Specify the threshold metrics
        detection_threshold    = 0.5
        verification_threshold = 0.8

        # Capture image and save it in input folder
        INPUT_PATH = os.path.join('application_data', 'input', 'input_image.jpg')
        ret, frame = self.capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1, minSize=(60, 60))

        # Iterate through the faces needed 
        for (x, y, w, h) in faces:
            
            # Crop the detected face region and resize to 250x250
            face_roi = frame[y:y + h, x:x + w]               
            cv2.imwrite(INPUT_PATH, cv2.resize(face_roi, (250,250)))


        results = []

        for image in os.listdir(os.path.join('application_data', 'verification_image')):
            # Load the validation and input image
            input_img      = self.pre_process(INPUT_PATH)
            validation_img = self.pre_process(os.path.join('application_data', 'verification_image', image))

            print(input_img)
            print(validation_img)

            # Make predictions          
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        # Detection threshold per image
        detection = np.sum(np.array(results) > detection_threshold)

        # Verification threshold
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_image')))
        verified = verification > verification_threshold

        self.label.text = 'Verified' if verified == True else 'Unverified'

        Logger.info(detection)
        Logger.info(verification)


        return results, verified





if __name__ == '__main__':
    VerificationApp().run()



