import cv2
import numpy as np
from tensorflow.keras.models import load_model
from facenet_pytorch import MTCNN

model = load_model('face_recognition_model.keras')
label_encoder = np.load('label_encoder.npy', allow_pickle=True)

mtcnn = MTCNN(keep_all=True)

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(img)
    faces = []
    if boxes is not None:
        for box in boxes:
            face = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            faces.append(face)
    return faces

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = preprocess_image(frame)
    for face in faces:
        face_resized = cv2.resize(face, (160, 160))
        face_normalized = face_resized / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)
        prediction = model.predict(face_expanded)
        predicted_class = np.argmax(prediction)
        label = label_encoder[predicted_class]
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
