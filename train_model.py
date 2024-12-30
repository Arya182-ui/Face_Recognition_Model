import cv2
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from facenet_pytorch import MTCNN
import os

dataset_dir = "./dataset"
mtcnn = MTCNN(keep_all=True)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(img)
    if boxes is not None:
        faces = []
        for box in boxes:
            face = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            faces.append(face)
        return faces
    return None

def load_images_from_dataset(dataset_dir):
    images = []
    labels = []
    label_encoder = LabelEncoder()
    for label_dir in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_dir)
        if os.path.isdir(label_path):
            label = label_dir
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                faces = preprocess_image(img_path)
                if faces:
                    for face in faces:
                        face_resized = cv2.resize(face, (160, 160))
                        images.append(face_resized)
                        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    labels_encoded = label_encoder.fit_transform(labels)
    return images, labels_encoded, label_encoder

images, labels_encoded, label_encoder = load_images_from_dataset(dataset_dir)
images = images / 255.0
np.save("label_encoder.npy", label_encoder.classes_)

def create_model(input_shape=(160, 160, 3)):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.fit(images, labels_encoded, epochs=10, batch_size=32, validation_split=0.2)
model.save("face_recognition_model.keras")
