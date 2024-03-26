from flask import Flask, render_template, request, jsonify
import cv2
import base64
import os
import shutil
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_DIR = 'modified_images'
app.config['UPLOAD_DIR'] = UPLOAD_DIR

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def apply_changes(image, crop_box=None, blur_radius=None, rotation=None, to_grayscale=False, detect_edges=False,
                        brightness=None, contrast=None, apply_sepia=False):
    if crop_box:
        x1, y1, x2, y2 = crop_box
        image = image[y1:y2, x1:x2]
    if blur_radius is not None and blur_radius > 0:
        image = cv2.GaussianBlur(image, (blur_radius, blur_radius), 0)
    if rotation:
        height, width = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation, 1)
        image = cv2.warpAffine(image, rotation_matrix, (width, height))
    if to_grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if detect_edges:
        image = cv2.Canny(image, 100, 200)
    if brightness is not None:
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    if contrast is not None:
        image = np.clip(image * contrast, 0, 255).astype(np.uint8)
    if apply_sepia:
        image = apply_sepia_filter(image)
    return image

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return image

def apply_sepia_filter(image):
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255)
    sepia_image = np.uint8(sepia_image)
    return sepia_image

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        filename = os.path.join(app.config['UPLOAD_DIR'], file.filename)
        file.save(filename)

        img = cv2.imread(filename)
        
        crop_box = None
        blur_radius = None
        rotation = None
        to_grayscale = False
        detect_edges = False
        brightness = None
        contrast = None
        apply_sepia = False

        if 'crop' in request.form and request.form['crop']:
            crop_data = request.form['crop'].split(',')
            crop_box = tuple(map(int, map(round, map(float, crop_data))))
        if 'blur' in request.form and request.form['blur']:
            blur_radius = int(request.form['blur'])
        if 'rotate' in request.form and request.form['rotate']:
            rotation = int(request.form['rotate'])
        if 'grayscale' in request.form:
            to_grayscale = True
        if 'edge_detection' in request.form:
            detect_edges = True
        if 'brightness' in request.form:
            brightness = float(request.form['brightness'])
        if 'contrast' in request.form:
            contrast = float(request.form['contrast'])
        if 'sepia' in request.form:
            apply_sepia = True if request.form['sepia'] == 'true' else False
        
        modified_img = apply_changes(img, crop_box, blur_radius, rotation, to_grayscale, detect_edges,
                                           brightness, contrast, apply_sepia)
        
        try:
            if 'channel' in request.form:
                channel = request.form['channel']
            if channel == 'red':
                selected_channel = modified_img[:, :, 2]  
            elif channel == 'blue':
                selected_channel = modified_img[:, :, 0]  
            elif channel == 'green':
                selected_channel = modified_img[:, :, 1]  
            else:
                selected_channel = modified_img  #
        
            modified_img = selected_channel

        except:
            pass

        try:
            img_with_faces = detect_faces(modified_img)
        except:
            img_with_faces = img

        return jsonify({'modified_img': encode_image(modified_img), 'img_with_faces': encode_image(img_with_faces)})

    return render_template('index.html')


@app.route('/clear_images', methods=['POST'])
def clear_images():
    try:
        shutil.rmtree(app.config['UPLOAD_DIR'])
        os.makedirs(app.config['UPLOAD_DIR'])
        message = 'Images folder cleared successfully'
    except Exception as e:
        message = f'Error clearing images folder: {str(e)}'
    return jsonify({'message': message})

if __name__ == '__main__':
    app.run(debug=True)
