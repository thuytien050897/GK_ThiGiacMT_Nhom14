from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
from werkzeug.utils import secure_filename
from utils import predict_digits

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
MODEL_PATH = 'mnist_cnn.h5'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            predictions, coords = predict_digits(filepath, MODEL_PATH)
            
            # Load original image
            img = cv2.imread(filepath)

            # Draw rectangles around the digits
            for (pred, (x, y, w, h)) in zip(predictions, coords):
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, str(pred), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            # Save the image with predictions
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
            cv2.imwrite(result_filepath, img)
            
            return render_template('index.html', predictions=predictions, image_url=result_filepath)
    return render_template('index.html', predictions=[], image_url=None)

if __name__ == '__main__':
    app.run(debug=True)
