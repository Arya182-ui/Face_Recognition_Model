import os
import numpy as np
import cv2
from facenet_pytorch import MTCNN
from sklearn.preprocessing import LabelEncoder

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
            faces.append(cv2.resize(face, (160, 160)))
        return faces
    return None

def load_images_from_dataset(dataset_dir):
    images, labels = [], []
    label_encoder = LabelEncoder()

    for label_dir in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_dir)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                faces = preprocess_image(img_path)
                if faces:
                    for face in faces:
                        images.append(face)
                        labels.append(label_dir)

    images = np.array(images)
    labels = np.array(labels)

    labels_encoded = label_encoder.fit_transform(labels)
    np.save("label_encoder.npy", label_encoder.classes_)
    return images, labels_encoded, label_encoder

images, labels_encoded, label_encoder = load_images_from_dataset(dataset_dir)
images = images / 255.0  
