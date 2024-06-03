import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bước 1: Làm mờ Gauss để giảm nhiễu
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Bước 2: Chuyển ảnh về ảnh nhị phân (đen và trắng) bằng phép ngưỡng
    _, bin_img = cv2.threshold(gray, 0.5 * np.max(gray), 255, cv2.THRESH_BINARY_INV)

    # Bước 3: Thực hiện các phép biến đổi hình thái để loại bỏ các điểm nhiễu nhỏ
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # Bước 4: Tìm các đường viền và xác định các đường viền là các chữ số dựa trên diện tích
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Bước 5: Tìm các hình chữ nhật giới hạn cho các đường viền tương ứng với các chữ số
    area_threshold = 30  # Điều chỉnh ngưỡng diện tích dựa trên ảnh
    rois = []
    for cnt in contours:
        if cv2.contourArea(cnt) < area_threshold:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = bin_img[y:y+h, x:x+w]
        rois.append((roi, (x, y, w, h)))

    # Sắp xếp ROIs dựa trên tọa độ x để duy trì thứ tự của các chữ số
    rois = sorted(rois, key=lambda r: r[1][0])

    return rois

def prepare_images(rois):
    new_imgs = []
    coords = []
    for roi, coord in rois:
        # Resize mỗi ROI thành kích thước 28x28
        negative = cv2.resize(roi, (28, 28))
        new_imgs.append(negative)
        coords.append(coord)

    x_new = np.array([img / 255.0 for img in new_imgs])
    x_new = x_new.reshape(-1, 28, 28, 1)  # Reshape cho phù hợp với mô hình
    return x_new, coords

def predict_digits(image_path, model_path):
    rois = preprocess_image(image_path)
    x_new, coords = prepare_images(rois)

    # Load model đã được huấn luyện
    loaded_model = load_model(model_path)

    # Dự đoán
    y_proba = loaded_model.predict(x_new)
    y_pred = np.argmax(y_proba, axis=1)

    return y_pred.tolist(), coords
