# Photo Categorizer 📸

An intelligent photo management tool that automatically analyzes your images and categorizes them based on the faces and objects detected within them. 

The project uses **DeepFace** for facial recognition and **YOLOv8** for object detection, ensuring that your gallery is neatly organized without taking up extra disk space.

## Features ✨

- **Facial Recognition**: Place reference photos of known people in the `known_faces` directory, and the script will automatically recognize them in your test images.
- **Object Detection**: Detects common objects (like cats, cars, couches, etc.) and categorizes photos accordingly.
- **Zero Storage Overhead**: Uses **hardlinks** instead of copying files. A single photo can appear in multiple category folders (e.g., `Eray`, `cat`, `couch`) while physically taking up space only once on your hard drive.
- **Batch Processing**: Drop as many photos as you want into the `test_images` folder and process them all in one go.

## Setup 🛠️

1. **Install Dependencies**:
   Make sure you have Python installed. Run the following command to install the required libraries:
   ```bash
   pip install deepface tf-keras ultralytics opencv-python-headless
   ```

2. **Prepare Directories**:
   The script relies on three main directories (it will create them automatically on the first run if they don't exist):
   - `known_faces/`: Put reference photos of people you want to identify here. Create a subfolder for each person (e.g., `known_faces/Name/photo.jpg`).
   - `test_images/`: Drop all the unstructured photos you want to categorize into this folder.
   - `categorized_photos/`: The script will automatically output the organized (hardlinked) photos here.

## Usage 🚀

To batch process all images inside the `test_images` folder, simply run the script:
```bash
python vision_os.py
```

Alternatively, you can process a single, specific image by passing its path as an argument:
```bash
python vision_os.py path/to/your_image.jpg
```

## How It Works 🧠

1. **Analysis Phase**:
   - The script first runs **DeepFace** to check if any of the faces from your `known_faces` database are present in the image.
   - It then runs **YOLOv8** to detect any general objects in the scene.
   - It compiles a unique list of all detected categories.

2. **Categorization Phase (Hardlinking)**:
   - For every category detected (e.g., `Eray` and `dog`), the script creates a folder inside `categorized_photos`.
   - It creates a **hardlink** of the original image inside those folders. This means the image is not duplicated, saving your SSD/HDD space, but it acts like a completely independent file in your OS file explorer.
