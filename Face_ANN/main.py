import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical


image_paths = [
    "Face_ANN/apj.jpg",
    "Face_ANN/bose.jpg",
    "Face_ANN/dr_br.jpg",
    "Face_ANN/gandhi.jpg",
    "Face_ANN/nehru.jpg"
]
names = ["APJ Abdul Kalam", 
         "Subhas Chandra Bose", 
         "B R Ambedkar", 
         "Mahatma Gandhi", 
         "Jawaharlal Nehru"]
X = []
y = []

import cv2
import numpy as np

X = []
y = []

for i, path in enumerate(image_paths):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))   # 64x64 = 4096
    img = img / 255.0                # normalize
    
    img = img.flatten()              # VERY IMPORTANT
    
    X.append(img)
    y.append(i)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)  # Label = 0,1,2,3,4



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(4096,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

X = np.array(X)
y = np.array(y)

y = to_categorical(y, 5)

print("X shape:", X.shape)
print("y shape:", y.shape)

model.fit(X, y, epochs=100, batch_size=1)


def predict_face(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = img.flatten().reshape(1, 4096)

    prediction = model.predict(img)
    person_id = np.argmax(prediction)
    confidence = np.max(prediction)

    print("Predicted Person:", names[person_id])
    print("Confidence:", confidence)


predict_face("Face_ANN/gandhi.jpg")