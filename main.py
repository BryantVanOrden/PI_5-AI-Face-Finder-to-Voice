import os
import requests
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import json
import pyttsx3
import threading
import hashlib

# ------------------ API Configurations ------------------
TESTING_MODE = True
APITOKEN = 'YOUR_API_TOKEN'  # Replace with your actual API token

# ------------------ Model Download Section ------------------
# We now use the landmark model from the patlevin repo.
MODEL_FILENAME = "face_landmark.tflite"
# This URL should point to the raw model file. Update if necessary.
MODEL_URL = "https://github.com/shaqian/tflite-models/raw/master/face_detection/face_detection_short_range.tflite"

def download_model_if_needed(model_path, model_url):
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Downloading from {model_url} ...")
        try:
            response = requests.get(model_url, stream=True)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Download complete.")
            else:
                print("Failed to download model. Status code:", response.status_code)
        except Exception as e:
            print("Error downloading model:", e)

download_model_if_needed(MODEL_FILENAME, MODEL_URL)
# ------------------ End Model Download Section ------------------

# Try to import tflite_runtime; fall back to TensorFlow Lite Interpreter if needed.
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    print("✅ Using tflite_runtime.")
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
    load_delegate = None  # Fallback: no delegate loading with TensorFlow
    print("✅ Using TensorFlow Lite fallback.")

# Global variables for tracking captured faces.
captured_face_ids = set()
capture_lock = threading.Lock()  # For thread-safe protection

# Cooldown (in seconds) to avoid rapid consecutive captures.
COOLDOWN = 5

# Directory to store images and data writeups.
OUTPUT_DIR = "captured_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize the text-to-speech engine.
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

def load_interpreter(model_path):
    """Load the TFLite model. Skip Edge TPU if not available."""
    try:
        # Only try delegate if explicitly wanted and available
        if load_delegate:
            try:
                interpreter = Interpreter(
                    model_path=model_path,
                    experimental_delegates=[load_delegate('libedgetpu.so.1')]
                )
                print("✅ Using Edge TPU delegate.")
            except OSError:
                print("⚠️ Edge TPU not available, using CPU.")
                interpreter = Interpreter(model_path=model_path)
        else:
            interpreter = Interpreter(model_path=model_path)
    except Exception as e:
        print("❌ Failed to load model:", e)
        raise e
    interpreter.allocate_tensors()
    return interpreter


def detect_faces(interpreter, frame):
    """
    Preprocess the frame, run inference, and return a list of detected faces.
    Each face is a tuple: (x1, y1, x2, y2, score, landmark_points)
      - (x1,y1,x2,y2): bounding box (pixel coordinates)
      - score: detection confidence
      - landmark_points: list of (x,y) tuples (assumed 5 points)
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Get input size (assumes shape: [1, height, width, 3])
    input_height, input_width = input_details[0]['shape'][1:3]

    # Resize and convert from BGR (OpenCV) to RGB.
    resized = cv2.resize(frame, (input_width, input_height))
    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb_frame, axis=0).astype(np.float32)
    # Normalize to [-1,1] if required by the model.
    input_data = (input_data - 127.5) / 127.5

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve outputs.
    # Assumptions based on the repo:
    #  - output_details[0]: bounding boxes, shape [1, N, 4] in normalized (ymin, xmin, ymax, xmax)
    #  - output_details[1]: scores, shape [1, N]
    #  - output_details[2]: landmarks, shape [1, N, 10] (5 points, each x,y)
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[1]['index'])[0]
    landmarks = interpreter.get_tensor(output_details[2]['index'])[0]

    detection_threshold = 0.5
    faces = []
    for i, score in enumerate(scores):
        if score > detection_threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            x1 = int(xmin * frame.shape[1])
            y1 = int(ymin * frame.shape[0])
            x2 = int(xmax * frame.shape[1])
            y2 = int(ymax * frame.shape[0])
            # Process landmarks (assumed to be normalized)
            lm = landmarks[i]  # vector of length 10.
            landmark_points = []
            for j in range(0, len(lm), 2):
                lx = int(lm[j] * frame.shape[1])
                ly = int(lm[j+1] * frame.shape[0])
                landmark_points.append((lx, ly))
            faces.append((x1, y1, x2, y2, score, landmark_points))
    return faces

def search_by_face(image_file):
    """
    Uses the FaceCheck API to search by face for the given image.
    Follows the API example from their website.
    Returns (error, results), where error is None if successful and
    results is a list of result items.
    """
    site = 'https://facecheck.id'
    headers = {'accept': 'application/json', 'Authorization': APITOKEN}
    with open(image_file, 'rb') as f:
        files = {'images': f, 'id_search': None}
        response = requests.post(site+'/api/upload_pic', headers=headers, files=files).json()

    if response.get('error'):
        return f"{response['error']} ({response.get('code', 'no code')})", None

    id_search = response.get('id_search')
    print(response.get('message', '') + ' id_search=' + id_search)
    json_data = {
        'id_search': id_search,
        'with_progress': True,
        'status_only': False,
        'demo': TESTING_MODE
    }

    while True:
        response = requests.post(site+'/api/search', headers=headers, json=json_data).json()
        if response.get('error'):
            return f"{response['error']} ({response.get('code', 'no code')})", None
        if response.get('output'):
            return None, response['output']['items']
        print(f'{response["message"]} progress: {response["progress"]}%')
        time.sleep(1)

class FaceCheckApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Check Application")
        
        # Panel for live video feed.
        self.video_panel = tk.Label(root)
        self.video_panel.pack(side="left", padx=10, pady=10)
        
        # Panel for the last captured image.
        self.captured_panel = tk.Label(root)
        self.captured_panel.pack(side="right", padx=10, pady=10)
        
        # Status label.
        self.status_label = tk.Label(root, text="Initializing...", font=("Arial", 14))
        self.status_label.pack(side="top", fill="x", padx=10, pady=10)
        
        # Log area.
        self.log_text = tk.Text(root, height=10)
        self.log_text.pack(side="bottom", fill="x", padx=10, pady=10)
        
        # Open the camera.
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log("Error: Could not open camera")
            return
        
        self.interpreter = load_interpreter(MODEL_FILENAME)
        self.last_capture_time = 0
        
        self.update_video()
    
    def log(self, message):
        """Append a message to the log and update the status label."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.status_label.config(text=message)
    
    def update_video(self):
        """Capture and process a frame, then update the video feed panel."""
        ret, frame = self.cap.read()
        if ret:
            faces = detect_faces(self.interpreter, frame)
            
            # Draw bounding boxes, scores, and landmarks.
            for (x1, y1, x2, y2, score, landmarks) in faces:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                for (lx, ly) in landmarks:
                    cv2.circle(frame, (lx, ly), 2, (0, 0, 255), -1)
            
            # Convert frame for Tkinter.
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_panel.imgtk = imgtk
            self.video_panel.config(image=imgtk)
            
            # Trigger processing if exactly one face is detected.
            if len(faces) == 1 and (time.time() - self.last_capture_time > COOLDOWN):
                self.last_capture_time = time.time()
                self.log("One face detected in good view! Capturing image...")
                threading.Thread(target=self.process_face, args=(frame.copy(),)).start()
            else:
                if len(faces) == 0:
                    self.log("Waiting for a good face...")
                elif len(faces) > 1:
                    self.log("Multiple faces detected. Waiting for one person.")
                    
        self.root.after(30, self.update_video)
    
    def process_face(self, frame):
        """Capture the face, send it to the FaceCheck API, and update the UI."""
        temp_filename = os.path.join(OUTPUT_DIR, "temp_capture.jpg")
        cv2.imwrite(temp_filename, frame)
        self.log("Sending image to FaceCheck API...")
        
        error, search_results = search_by_face(temp_filename)
        if error:
            self.log("Error from API: " + error)
            os.remove(temp_filename)
            return
        if not search_results or len(search_results) == 0:
            self.log("No search results returned from API.")
            os.remove(temp_filename)
            return
        
        # Use the top result's URL to create a unique identifier.
        top_result = search_results[0]
        face_id = hashlib.md5(top_result["url"].encode('utf-8')).hexdigest()
        
        with capture_lock:
            if face_id in captured_face_ids:
                self.log("Face already captured. Skipping.")
                speak("Face already captured.")
                os.remove(temp_filename)
                return
            else:
                captured_face_ids.add(face_id)
        
        # Rename the temporary image file to include the face ID.
        image_filename = os.path.join(OUTPUT_DIR, f"face_{face_id}.jpg")
        os.rename(temp_filename, image_filename)
        
        # Create a writeup file linking the image and API response.
        writeup_filename = os.path.join(OUTPUT_DIR, f"face_{face_id}.txt")
        try:
            with open(writeup_filename, 'w') as f:
                f.write("Face Check API Data:\n")
                f.write(json.dumps(search_results, indent=2))
                f.write("\n\n")
                f.write(f"Image File: {image_filename}\n")
            self.log(f"Saved data to {writeup_filename}")
        except Exception as e:
            self.log("Error writing file: " + str(e))
        
        # Update the captured image display.
        captured_img = cv2.imread(image_filename)
        if captured_img is not None:
            captured_img_rgb = cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(captured_img_rgb)
            imgtk2 = ImageTk.PhotoImage(image=pil_img)
            self.captured_panel.imgtk = imgtk2
            self.captured_panel.config(image=imgtk2)
        
        summary = f"New face captured with top match score: {top_result.get('score', 'N/A')}."
        speak(summary)
        self.log(summary)
    
    def on_closing(self):
        """Clean up resources on exit."""
        self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceCheckApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
