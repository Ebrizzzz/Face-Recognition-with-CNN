import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
from keras_facenet import FaceNet
import joblib
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the trained model and encoders
model = joblib.load('Saved Models/FaceNet model/face_classifier.pkl')
in_encoder = joblib.load('Saved Models/FaceNet model/in_encoder.pkl')
out_encoder = joblib.load('Saved Models/FaceNet model/out_encoder.pkl')

# Initialize FaceNet for embedding extraction
embedder = FaceNet()

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

    # Use keras_facenet to extract detections
    detections = embedder.extract(filename, threshold=0.95)

    if not detections:
        print('No face detected')
        messagebox.showerror("Error", "No face detected.")
        return

    # Load the image using OpenCV to draw rectangles
    image = cv2.imread(filename)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Initialize figure for plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image_rgb)
    ax.axis('off')

    # Initialize a list to store the predicted names and probabilities
    predictions = []

    # Iterate over each detected face
    for detection in detections:
        # Extract the embedding
        embedding = detection['embedding']
        embedding_norm = in_encoder.transform([embedding])

        # Predict the class and probability
        yhat_class = model.predict(embedding_norm)
        yhat_prob = model.predict_proba(embedding_norm)

        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_name = out_encoder.inverse_transform([class_index])[0]

        # Determine color for rectangle based on class
        color = (0, 255, 0)  # Green color for known faces
        # here We define our threshold for face reconition.
        if class_probability <= 85:
            predict_name = 'Unknown'
            color = (255, 0, 0)  # Red color for unknown faces
            class_probability = 0.0  # Set probability to 0 for unknown faces

        # Draw rectangle around the face
        x1, y1, width, height = detection['box']
        x2, y2 = x1 + width, y1 + height
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), color, 2)

        # Display predicted name
        text = f'{predict_name} ({class_probability:.3f}%)' if predict_name != 'Unknown' else f'{predict_name}'
        # Adjust text position to be visible even on small faces
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20

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
