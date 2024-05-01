import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpeg") or filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            images.append(  cv2.resize( cv2.imread(img_path), (300,300) ).flatten()  )
    return images
        
parrot_images = load_images("parrot")
pigeon_images = load_images("pigeon")


""" 1 ler parrot"""
parrot_ = np.ones(len(parrot_images))
pigeon_ = np.zeros(len(pigeon_images))

x = parrot_images+pigeon_images

y = np.hstack((parrot_,pigeon_))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)

# Modeli değerlendir
predictions = model.predict(x_test)

# Metrikleri hesapla
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

# Sonuçları ekrana yazdır
print("Model Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1,"\n")


test_dosyası = load_images("test")
a = model.predict(test_dosyası)
print(a)

