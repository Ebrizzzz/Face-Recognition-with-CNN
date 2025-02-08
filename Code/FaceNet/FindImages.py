import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import joblib
from keras_facenet import FaceNet
import cv2
from tkinter import ttk

# Load the model and encoders
model = joblib.load('Saved Models/FaceNet model/face_classifier.pkl')
in_encoder = joblib.load('Saved Models/FaceNet model/in_encoder.pkl')
out_encoder = joblib.load('Saved Models/FaceNet model/out_encoder.pkl')
embedder = FaceNet()

# Get the list of recognizable people
recognizable_people = list(out_encoder.classes_)


def process_images():
    selected_people = [listbox.get(i) for i in listbox.curselection()]
    if not selected_people:
        messagebox.showerror("Error", "No people selected.")
        return

    folder_path = filedialog.askdirectory()

    if not folder_path:
        messagebox.showerror("Error", "No folder selected.")
        return

    # Create a new folder named after the selected people if it doesn't already exist
    save_folder_name = '-'.join(selected_people)
    save_folder_path = os.path.join('filtered_images', save_folder_name)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    processed_count = 0
    matched_count = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(folder_path, filename)
            detections = embedder.extract(file_path, threshold=0.95)
            recognized_faces = []

            for detection in detections:
                embedding = detection['embedding']
                embedding_norm = in_encoder.transform([embedding])
                yhat_class = model.predict(embedding_norm)
                yhat_prob = model.predict_proba(embedding_norm)
                class_index = yhat_class[0]
                class_probability = yhat_prob[0, class_index] * 100
                predict_name = out_encoder.inverse_transform([class_index])[0]

                if class_probability > 85:
                    recognized_faces.append(predict_name)
                else:
                    recognized_faces.append('Unknown')

            if set(selected_people).issubset(set(recognized_faces)):
                shutil.copy(file_path, os.path.join(save_folder_path, filename))
                matched_count += 1
            
            processed_count += 1
            status_label.config(text=f"Processed {processed_count} images. Matched {matched_count} images.")
            root.update_idletasks()
    
    messagebox.showinfo("Completed", f"Processing completed. {matched_count} images matched the criteria and were copied to {save_folder_path}.")

# Create the main window
root = tk.Tk()
root.title("Select People for Image Filtering")
root.configure(bg="#2e2e2e")

frame = tk.Frame(root, bg="#2e2e2e")
frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

label = tk.Label(frame, text="Select the people you want to filter images for:", bg="#2e2e2e", fg="white", font=("Helvetica", 16))
label.pack(pady=10)

listbox_frame = tk.Frame(frame, bg="#2e2e2e")
listbox_frame.pack(pady=5)
listbox_scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL)
listbox = tk.Listbox(listbox_frame, selectmode=tk.MULTIPLE, height=10, bg="#404040", fg="white", font=("Helvetica", 14), yscrollcommand=listbox_scrollbar.set)
listbox_scrollbar.config(command=listbox.yview)
listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
listbox.pack(side=tk.LEFT, fill=tk.BOTH)

for person in recognizable_people:
    listbox.insert(tk.END, person)

button = tk.Button(frame, text="Select Folder and Process Images", command=process_images, font=("Helvetica", 14), bg="#007acc", fg="white", width=30)
button.pack(pady=20)



status_label = tk.Label(frame, text="Status: Waiting for user input...", fg="white", bg="#2e2e2e", font=("Helvetica", 14))
status_label.pack(pady=10)

root.mainloop()
