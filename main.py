import numpy as np
import cv2
from tensorflow.keras import layers, models 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D 
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Supaya log TensorFlow tidak mengganggu

EMOTIONS = ['Angry', 'Happy', 'Neutral', 'Suprise']
IMG_SIZE = 48

# Load model CNN
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),  # Mengurangi overfitting
    Dense(4, activation='softmax')  # Sesuaikan jumlah kelas
])


modelss = 'higgest_accuracy.h5'
# Pastikan model memiliki bobot yang benar  
try:
    model.load_weights(modelss)
except:
    print(f"Error: File '{modelss}' tidak ditemukan.")
    exit()

# Mencegah penggunaan OpenCL untuk mengurangi noise log
cv2.ocl.setUseOpenCL(False)

# Kamus label emosi
emotion_dict = {0 :'Angry', 1:'Happy',2: 'Neutral',3: 'Suprise'}

# Load Haar Cascade bawaan OpenCV
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
facecasc = cv2.CascadeClassifier(face_cascade_path)

if facecasc.empty():
    print("Error: Haar Cascade tidak dapat dimuat. Periksa path file XML.")
    exit()

# Buka webcam (gunakan CAP_DSHOW untuk Windows agar lebih stabil)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Webcam tidak dapat diakses.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Gambar kotak di wajah
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)

        # Potong wajah dan ubah ukuran sesuai dengan model
        roi_color = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(roi_color, (224, 224))  # Model menerima input (224, 224, 3)
        input_img = np.expand_dims(resized_img, axis=0) / 255.0  # Normalisasi

        # Prediksi emosi
        prediction = model.predict(input_img)
        maxindex = int(np.argmax(prediction))
        
        # Tampilkan teks emosi
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Tampilkan output
    cv2.imshow('Output', cv2.resize(frame, (1000, 600), interpolation=cv2.INTER_CUBIC))

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
