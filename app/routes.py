from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from collision_detection import process_video

main = Blueprint('main', __name__)

UPLOAD_FOLDER = 'uploaded_videos'  # Specify your upload directory
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Process the video using your detection functions
        collision_result, road_signs = process_video(filepath, roi=100)  # Adjust ROI as needed

        return render_template('index.html', collision_result=collision_result, road_signs=road_signs)
