import os
import cv2
from mtcnn import MTCNN
from PIL import Image
import matplotlib.pyplot as plt

def detect_and_crop_faces(image_path, output_dir):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        return

    # Convert the image to RGB (OpenCV loads images in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize MTCNN detector
    detector = MTCNN()
    
    # Detect faces in the image
    results = detector.detect_faces(image_rgb)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot the image with detected faces
    plt.imshow(image_rgb)
    ax = plt.gca()
    
    # Loop through all detected faces
    for i, result in enumerate(results):
        # Get the bounding box coordinates
        x, y, width, height = result['box']
        x, y = abs(x), abs(y)
        
        # Draw rectangle around the face
        rect = plt.Rectangle((x, y), width, height, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Crop the face from the image
        cropped_face = image_rgb[y:y+height, x:x+width]
        
        # Convert the cropped face to an Image object
        face_image = Image.fromarray(cropped_face)
        
        # Save the cropped face image to the output directory
        face_image_path = os.path.join(output_dir, f"face_{i+1}.jpg")
        face_image.save(face_image_path)
        
        print(f"Saved cropped face {i+1} to {face_image_path}")
    
    # Display the plot
    plt.axis('off')
    plt.show()

# Example usage
image_path = 'etr.jpg'
output_dir = 'cropped_faces'
detect_and_crop_faces(image_path, output_dir)
