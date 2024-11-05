import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from preprocess import load_data, preprocess_data

# Đường dẫn tới tập dữ liệu và các tham số cơ bản
data_dir = "E:\\BaiCuoiKiAi\\GTSRB\\Final_Training\\Images"  # Thư mục chứa dữ liệu
image_size = 30
classes = 43  # Số lớp biển báo

# Bước 1: Tải và xử lý dữ liệu
data, labels = load_data(data_dir, image_size, classes)
X_train, X_test, y_train, y_test = preprocess_data(data, labels, classes)

# Bước 2: Xây dựng mô hình CNN
model = Sequential()

# Lớp Conv2D đầu tiên với MaxPooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Lớp Conv2D thứ hai với MaxPooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Lớp Conv2D thứ ba với MaxPooling
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Lớp Flatten để chuyển từ ma trận 2D thành vector 1D
model.add(Flatten())

# Lớp Fully Connected với Dropout để tránh overfitting
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Lớp output với số lớp tương ứng số lượng biển báo, activation là softmax
model.add(Dense(classes, activation='softmax'))

# Bước 3: Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Bước 4: Huấn luyện mô hình
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Bước 5: Đánh giá mô hình trên tập kiểm thử
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Độ chính xác trên tập kiểm thử: {test_acc * 100:.2f}%")

# Lưu mô hình sau khi huấn luyện (tùy chọn)
model.save("traffic_sign_cnn_model.h5")
