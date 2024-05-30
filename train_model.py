import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from math import ceil


# Bước 1: Thu thập dữ liệu
# Tải dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Bước 2: Tiền xử lý dữ liệu
# Reshape dữ liệu về kích thước (28, 28, 1) và chuẩn hóa giá trị pixel về [0, 1]
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Chuyển đổi nhãn sang dạng one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

from tensorflow.keras.layers import Dropout

# Bước 3: Xây dựng mô hình CNN với lớp Dropout
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),  # Thêm một lớp Conv2D với 128 bộ lọc
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Bước 4: Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Bước 5: Tăng cường dữ liệu
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
datagen.fit(x_train)
# Bước 6: Huấn luyện mô hình với dữ liệu tăng cường
model.fit(datagen.flow(x_train, y_train, batch_size=300), steps_per_epoch = ceil(len(x_train) / 300), epochs=20, validation_data=(x_test, y_test))

# Bước 7: Đánh giá mô hình
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {test_loss:.4f}')
print(f'Test accuracy: {test_accuracy:.4f}')

# Kiểm tra nếu test loss và test accuracy đều được tính toán thành công
if test_loss is not None and test_accuracy is not None:
    print("The model has successfully trained")

# Bước 8: Lưu mô hình đã huấn luyện
model.save('cnn_model.keras')
