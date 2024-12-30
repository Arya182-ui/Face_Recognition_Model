import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model("face_recognition_model.keras")
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("label_encoder.npy")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (160, 160)) / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)
        prediction = model.predict(face_resized)
        predicted_label_index = np.argmax(prediction)
        predicted_name = label_encoder.inverse_transform([predicted_label_index])[0]
        cv2.putText(frame, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
