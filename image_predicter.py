import cv2
import numpy as np

# Load image
img = cv2.imread('path_to_image.jpg')
face_resized = cv2.resize(img, (160, 160))
face_resized = face_resized / 255.0  # Normalize

# Add batch dimension
face_resized = np.expand_dims(face_resized, axis=0)

# Predict the label
prediction = model.predict(face_resized)
predicted_label_index = np.argmax(prediction)
predicted_name = label_encoder.inverse_transform([predicted_label_index])[0]

print("Predicted Name:", predicted_name)
