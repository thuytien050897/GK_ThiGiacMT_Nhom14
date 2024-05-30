from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load mô hình đã huấn luyện
model = load_model('cnn_model.keras')

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_imgs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Loại bỏ các contour có diện tích nhỏ hơn ngưỡng
        area = cv2.contourArea(cnt)
        if area > 50:  # Chỉ chấp nhận các contour có diện tích đủ lớn
            digit_img = img[y:y+h, x:x+w]
            digit_img = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
            digit_img = digit_img.astype('float32') / 255
            digit_img = np.expand_dims(digit_img, axis=-1)
            digit_imgs.append(digit_img)


    # # In ra ảnh đã tiền xử lý
    # for digit_img in digit_imgs:
    #     cv2.imshow('Preprocessed Image', digit_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # # In ra contours
    # img_with_contours = cv2.drawContours(cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR), contours, -1, (0, 255, 0), 2)
    # cv2.imshow('Image with Contours', img_with_contours)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return digit_imgs, contours

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            digit_imgs, contours = preprocess_image(file_path)
            prediction = ""
            for digit_img in digit_imgs:
                digit_img = np.expand_dims(digit_img, axis=0)
                pred = model.predict(digit_img)
                digit = np.argmax(pred, axis=1)[0]
                prediction += str(digit)

            # In ra dự đoán của từng chữ số
            # prediction = prediction[::-1]
            print("Predictions:", prediction)

            return render_template('index.html', prediction=prediction)

    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)