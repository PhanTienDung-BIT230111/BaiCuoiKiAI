from preprocess import load_data, preprocess_data
from model import build_model, train_model
from predict import predict_sign
from realtime_detection import realtime_detection

data_dir = "E:\\BaiCuoiKiAi\\GTSRB\\Final_Training\\Images"
image_size = 30
classes = 43

# Load và xử lý dữ liệu
data, labels = load_data(data_dir, image_size, classes)
X_train, X_test, y_train, y_test = preprocess_data(data, labels, classes)

# Xây dựng và huấn luyện mô hình
model = build_model(image_size, classes)
model = train_model(model, X_train, y_train, X_test, y_test)

# Dự đoán từ một hình ảnh tĩnh
image_path = "OIP.jpg"
predicted_class = predict_sign(model, image_path)
print(f"Biển báo dự đoán: {predicted_class}")

# Nhận diện biển báo theo thời gian thực
realtime_detection(model, image_size)
