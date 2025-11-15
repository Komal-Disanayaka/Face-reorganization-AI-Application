# Face Recognition Web Application using Flask
# ==============================================

import os
import numpy as np
import imutils
import pickle
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import base64

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_in_production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models globally
print("[INFO] Loading Face Detector...")
protoPath = os.path.join('face_detection_model', 'deploy.prototxt')
modelPath = os.path.join('face_detection_model', 'res10_300x300_ssd_iter_140000.caffemodel')
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] Loading Face Embedder...")
embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

print("[INFO] Loading Trained Face Recognition Model...")
recognizer = pickle.loads(open('output/recognizer_tuned.pickle', "rb").read())
le = pickle.loads(open('output/le.pickle', "rb").read())

print("[INFO] Models loaded successfully!")


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def recognize_faces(image_path):
    """
    Perform face recognition on the given image
    Returns: processed image with annotations and list of recognized faces
    """
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        return None, []
    
    # Resize image for faster processing
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    
    # Construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    
    # Detect faces
    detector.setInput(imageBlob)
    detections = detector.forward()
    
    recognized_faces = []
    
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections
        if confidence > 0.5:
            # Compute bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Extract face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            
            # Ensure face is sufficiently large
            if fW < 20 or fH < 20:
                continue
            
            # Create blob for face ROI
            faceBlob = cv2.dnn.blobFromImage(
                face, 1.0 / 255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False
            )
            
            # Generate face embedding
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            
            # Perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            
            # Store recognition result
            recognized_faces.append({
                'name': name,
                'confidence': float(proba * 100),
                'bbox': (int(startX), int(startY), int(endX), int(endY))
            })
            
            # Draw bounding box and label on the image
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            # Use different colors based on confidence
            color = (0, 255, 0) if proba > 0.7 else (0, 165, 255)  # Green if high confidence, orange if lower
            
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            cv2.putText(image, text, (startX, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return image, recognized_faces


@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and perform face recognition"""
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename or 'upload.jpg')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Perform face recognition
        result_image, recognized_faces = recognize_faces(filepath)
        
        if result_image is None:
            flash('Error processing image', 'error')
            return redirect(url_for('index'))
        
        # Save the result image
        result_filename = 'result_' + filename
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_filepath, result_image)
        
        # Pass results to template
        return render_template('result.html',
                             original_image=filename,
                             result_image=result_filename,
                             recognized_faces=recognized_faces,
                             total_faces=len(recognized_faces))
    else:
        flash('Invalid file type. Please upload an image file (png, jpg, jpeg, gif)', 'error')
        return redirect(url_for('index'))


@app.route('/clear')
def clear_uploads():
    """Clear uploaded files"""
    try:
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        flash('All uploads cleared successfully', 'success')
    except Exception as e:
        flash(f'Error clearing uploads: {str(e)}', 'error')
    
    return redirect(url_for('index'))


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Face Recognition Web Application")
    print("="*50)
    print("\nServer starting...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
