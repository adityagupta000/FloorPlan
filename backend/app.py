from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from inference import run_segmentation
from obj_generator import generate_obj_from_mask

app = Flask(__name__)
CORS(app)

# Base directory (backend folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folders (always valid regardless of where app is launched)
UPLOAD_FOLDER = os.path.join(BASE_DIR, '../uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, '../outputs')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -----------------------------------------------------
# Health check
# -----------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'Backend running'
    }), 200


# -----------------------------------------------------
# Upload floorplan
# -----------------------------------------------------
@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image field in request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        return jsonify({
            'message': 'Image uploaded successfully',
            'filename': filename,
            'filepath': save_path.replace('\\', '/')
        }), 200

    return jsonify({
        'error': 'Invalid file type. Allowed: PNG, JPG, JPEG'
    }), 400


# -----------------------------------------------------
# Segmentation API
# -----------------------------------------------------
@app.route('/api/segment', methods=['POST'])
def segment_image():
    data = request.get_json()

    if not data or 'filename' not in data:
        return jsonify({'error': 'Missing filename'}), 400

    filename = data['filename']
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(input_path):
        return jsonify({'error': 'Image file not found'}), 404

    try:
        mask_filename = f"mask_{filename}"
        mask_path = os.path.join(app.config['OUTPUT_FOLDER'], mask_filename)

        success = run_segmentation(input_path, mask_path)

        if success:
            return jsonify({
                'message': 'Segmentation completed',
                'mask_filename': mask_filename,
                'mask_path': mask_path.replace('\\', '/')
            }), 200
        else:
            return jsonify({'error': 'Segmentation failed'}), 500

    except Exception as e:
        return jsonify({
            'error': 'Segmentation error',
            'details': str(e)
        }), 500


# -----------------------------------------------------
# 3D Model Generation API
# -----------------------------------------------------
@app.route('/api/generate3d', methods=['POST'])
def generate_3d_model():
    data = request.get_json()

    if not data or 'mask_filename' not in data:
        return jsonify({'error': 'Missing mask_filename'}), 400

    mask_filename = data['mask_filename']
    mask_path = os.path.join(app.config['OUTPUT_FOLDER'], mask_filename)

    if not os.path.exists(mask_path):
        return jsonify({'error': 'Mask file not found'}), 404

    try:
        obj_filename = mask_filename.replace('mask_', '').rsplit('.', 1)[0] + '.obj'
        obj_path = os.path.join(app.config['OUTPUT_FOLDER'], obj_filename)

        success = generate_obj_from_mask(mask_path, obj_path)

        if success:
            return jsonify({
                'message': '3D model generated',
                'obj_filename': obj_filename,
                'obj_path': obj_path.replace('\\', '/')
            }), 200
        else:
            return jsonify({'error': 'OBJ generation failed'}), 500

    except Exception as e:
        return jsonify({
            'error': '3D generation error',
            'details': str(e)
        }), 500


# -----------------------------------------------------
# File Download
# -----------------------------------------------------
@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)

    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404

    return send_file(filepath, as_attachment=True, download_name=filename)


# -----------------------------------------------------
# Main
# -----------------------------------------------------
if __name__ == '__main__':
    print("\n==============================")
    print(" Starting Flask server...")
    print(" URL: http://localhost:5000")
    print(" Ready to process floor plans!")
    print("==============================\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
