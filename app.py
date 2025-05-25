from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
from flask_cors import CORS
import time
import socket                                                  # for getting ip add

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Set an upload folder and allowed extensions (optional)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Custom Yolo Model
model = YOLO("best.pt")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_local_ip():
    try:
        # Create a socket connection to a dummy address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Google's public DNS
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        return f"Could not determine local IP: {e}"


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file part in the request'}), 400
    
    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # rename filename
        # parts = filename.rsplit('.', 1)
        # new_filename = f"prediction.{parts[1]}"


        # filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # get result
        results = model.predict(filepath, save=True, project="static", exist_ok=True, conf=0.4)

        for result in results:
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]    # get kakanin_names of each box
            confs = result.boxes.conf.tolist()  # get confidences 
    
        rounded_confs = [round(num, 4) for num in confs]    # round confidence results

        # Dictionary to store highest confidence per class
        result_dict = {}

        for name, conf in zip(names, rounded_confs):
            name_lower = name.lower()
            if name_lower not in result_dict or conf > result_dict[name_lower]:
                result_dict[name_lower] = conf

        # Convert to list of dictionaries
        result_list = [{'name': name, 'conf': conf} for name, conf in result_dict.items()]

        print(result_list)
        
        # os.remove(filepath)

        # removes the file extension from the filename
        filename = os.path.splitext(filename)[0]
        filename = f"{filename}.jpg"

        if not names:
            # return jsonify({'error': 'No Kakanin Detected'}), 200
            return jsonify({
                'result_list': [],
                'image_url': f'http://{get_local_ip()}:5000/static/predict/{filename}'
            }), 200
        

        # Wait until the saved prediction image appears
        full_output_path = os.path.join("static", "predict", filename)

        timeout = 60  # seconds
        waited = 0
        while not os.path.exists(full_output_path) and waited < timeout:
            print("file not exists")
            time.sleep(0.1)
            waited += 0.1
        
        print(f'local ip: {get_local_ip()}')

        return jsonify({
            'result_list': result_list,
            # 'image_url': f'http://192.168.107.180:5000/static/predict/{new_filename}'
            # 'image_url': f'http://192.168.107.180:5000/static/predict/{filename}'
            'image_url': f'http://{get_local_ip()}:5000/static/predict/{filename}'

        }), 200

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
    