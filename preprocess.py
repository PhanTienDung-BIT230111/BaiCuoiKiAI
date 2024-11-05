import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


def load_data(data_dir, image_size=30, classes=43):
    data = []
    labels = []

    for i in range(classes):
        path = os.path.join(data_dir, f'{i:05d}')  # Đường dẫn tới thư mục ảnh
        images = os.listdir(path)
        for img_name in images:
            try:
                img_path = os.path.join(path, img_name)
                if img_name.endswith(('.ppm', '.png', '.jpg')):  # Kiểm tra định dạng file ảnh
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"Bỏ qua ảnh bị hỏng: {img_name}")
                        continue
                    image = cv2.resize(image, (image_size, image_size))
                    data.append(image)
                    labels.append(i)
            except Exception as e:
                print(f"Lỗi khi tải ảnh {img_name}: {e}")

    data = np.array(data)
    labels = np.array(labels)
    return data, labels


def preprocess_data(data, labels, classes=43):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    X_train = X_train / 255.0  # Chuẩn hóa dữ liệu
    X_test = X_test / 255.0
    y_train = to_categorical(y_train, classes)  # Chuyển đổi nhãn thành dạng one-hot
    y_test = to_categorical(y_test, classes)
    return X_train, X_test, y_train, y_test


# Khởi tạo mô hình
def create_model(image_size, classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    return model


# Sử dụng các hàm này trong mã chính của bạn
data_dir = "E:\\BaiCuoiKiAi\\GTSRB\\Final_Training\\Images"
data, labels = load_data(data_dir)
X_train, X_test, y_train, y_test = preprocess_data(data, labels)

# Tạo và biên dịch mô hình
model = create_model(image_size=30, classes=43)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Lưu mô hình dưới định dạng Keras mới
model.save('my_model.keras')
