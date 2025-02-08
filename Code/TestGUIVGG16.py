import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the face recognition model
vgg_model = load_model('Saved Models/vgg16_Model.keras')

# Class names (modify these to match your dataset classes)
class_names = ['Henry Cavill', 'Natalie Portman', 'Robert Dawney Junior']

# Initialize MTCNN for face detection
detector = MTCNN()

# Keep a reference to the current canvas
current_canvas = None

def upload_image():
    filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if filename:
        img = Image.open(filename)
        img.thumbnail((400, 400)) 
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        predict_image(filename)  

def predict_image(filename):
    global current_canvas

    # Destroy the previous canvas if it exists
    if current_canvas:
        current_canvas.get_tk_widget().destroy()

    # Load the image using OpenCV to draw rectangles
    image = cv2.imread(filename)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Use MTCNN to detect faces
    detections = detector.detect_faces(image_rgb)

    if len(detections) == 0:
        print('No face detected')
        messagebox.showerror("Error", "No face detected.")
        return

    # Initialize figure for plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image_rgb)
    ax.axis('off')

    # Initialize a list to store the predicted names and probabilities
    predictions = []

    # Iterate over each detected face
    for detection in detections:
        x, y, width, height = detection['box']
        face = image_rgb[y:y+height, x:x+width]
        face_resized = cv2.resize(face, (224, 224))
        face_array = np.expand_dims(face_resized, axis=0) / 255.0  # Normalize the image

        # Predict the class
        predictions_probs = vgg_model.predict(face_array)[0]
        class_index = np.argmax(predictions_probs)
        class_probability = predictions_probs[class_index] * 100
        predict_name = class_names[class_index]

        # Determine color for rectangle based on class
        color = (0, 255, 0)  # Green color for known faces
        if class_probability <= 85:
            predict_name = 'Unknown'
            color = (255, 0, 0)  # Red color for unknown faces
            class_probability = 0.0  # Set probability to 0 for unknown faces

        # Draw rectangle around the face
        x2, y2 = x + width, y + height
        cv2.rectangle(image_rgb, (x, y), (x2, y2), color, 2)

        # Display predicted name
        text = f'{predict_name} ({class_probability:.3f}%)' if predict_name != 'Unknown' else f'{predict_name}'
        text_x = x
        text_y = y - 10 if y - 10 > 10 else y + 20

        cv2.putText(image_rgb, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Append the predicted name and probability to the list
        predictions.append(f'{predict_name}: {class_probability:.2f}%')

    # Display the image with rectangles and predictions
    ax.imshow(image_rgb)
    ax.set_title('Face Recognition')

    # Display the predicted names and probabilities as text below the image
    fig.text(0.5, 0.01, '\n'.join(predictions), ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # Embed the plot in the Tkinter window
    current_canvas = FigureCanvasTkAgg(fig, master=root)
    current_canvas.draw()
    current_canvas.get_tk_widget().pack(pady=20)

# Create the main window
root = tk.Tk()
root.title("Face Recognition")
root.configure(bg="black")  

# Create a frame to hold the upload button and label
frame = tk.Frame(root, bg="black")
frame.pack(expand=True)

# Create widgets
upload_button = tk.Button(frame, text="Upload Image", command=upload_image, font=("Helvetica", 14), bg="green", fg="white")
upload_label = tk.Label(frame, text="Click the button to upload", bg="black", fg="white", font=("Helvetica", 12))
image_label = tk.Label(root, bg="black")  # Display the uploaded image

# Arrange widgets in the frame
upload_label.pack(pady=10)
upload_button.pack(pady=5)
image_label.pack(pady=20)

root.mainloop()
