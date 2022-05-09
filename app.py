from flask import Flask,render_template, Response
import cv2

from deepface import DeepFace
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
app = Flask(__name__)
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        if cv2.waitKey(2) & 0xFF == ord('q'):
               break
        success, frame = cap.read()  # read the camera frame
        result = DeepFace.analyze(   frame , enforce_detection=False,actions=['emotion'])
        if not success:
            break
        else:
            small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray,1.1,4)
            for x,y,w,h in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,result['dominant_emotion'],(50,50),font,3,(0,0,255),2,cv2.LINE_4)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=="__main__":
    app.run(debug=True,port=8000)