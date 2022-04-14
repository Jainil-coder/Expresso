import imp
from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np
from keras.models import model_from_json  
from keras.preprocessing import image  
# model = model_from_json(open("fer.json", "r").read())  #load model  

# model.load_weights('fer.h5')                            #load weights  

# face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# camera = cv2.VideoCapture(0)

# app = Flask(__name__)
# def gen_frames():                                       # generate frame by frame from camera
#     while True:
#         # Capture frame by frame
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        
#             faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)  
            
        
#             for (x,y,w,h) in faces_detected:
                
#                 cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)  
#                 roi_gray=gray_img[y:y+w,x:x+h]          #cropping region of interest i.e. face area from  image  
#                 roi_gray=cv2.resize(roi_gray,(48,48))  
#                 img_pixels = image.img_to_array(roi_gray)  
#                 img_pixels = np.expand_dims(img_pixels, axis = 0)  
#                 img_pixels /= 255  
        
#                 predictions = model.predict(img_pixels)  
        
#                 max_index = np.argmax(predictions[0])   #find max indexed array
        
#                 emotions = [ 'surprise', 'neutral','angry', 'disgust', 'fear', 'happy', 'sad']  
#                 predicted_emotion = emotions[max_index]  
                
#                 cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)  
        
#             resized_img = cv2.resize(frame, (1000, 700)) 

model = model_from_json(open("fer.json", "r").read())
model.load_weights('fer(1).h5')


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)
app = Flask(__name__)
def gen_frames():  

    while True:
        ret,frame=cap.read()# captures frame and returns boolean value and captured image
        if not ret:
            break
        else:
            gray_img= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


            for (x,y,w,h) in faces_detected:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)
                roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
                roi_gray=cv2.resize(roi_gray,(48,48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)

                #find max indexed array
                max_index = np.argmax(predictions[0])

                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                predicted_emotion = emotions[max_index]

                cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            resized_img = cv2.resize(frame, (1000, 700)) 
                
            ret, buffer = cv2.imencode('.jpg', frame)
                
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)