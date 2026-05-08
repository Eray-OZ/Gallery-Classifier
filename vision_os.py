import os
import sys
import shutil
from deepface import DeepFace
from ultralytics import YOLO

# Initialize YOLO model (will be downloaded automatically on first run if missing)
# Using yolov8s.pt (small) instead of nano for better accuracy and fewer hallucinations
yolo_model = YOLO("yolov8s.pt")

# Configure these paths before running
KNOWN_FACES_DIR = "known_faces"       # Directory containing subfolders with known people's faces
CATEGORIZED_DIR = "categorized_photos" # Where the categorized links will be created
TEST_IMAGES_DIR = "test_images"        # Directory containing images to process

# Minimum confidence required for YOLO object detection (0.0 to 1.0)
# 0.25 is the YOLO default. We lower it back so it catches the dog on the couch.
YOLO_CONFIDENCE_THRESHOLD = 0.25

def predict_categories(file_path: str):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return []

    print(f"Analyzing '{file_path}'...")
    detected_categories = set()
    
    # 1. Face Recognition using DeepFace
    if os.path.exists(KNOWN_FACES_DIR) and os.listdir(KNOWN_FACES_DIR):
        try:
            # enforce_detection=False allows DeepFace to not crash if no face is found
            dfs = DeepFace.find(img_path=file_path, db_path=KNOWN_FACES_DIR, enforce_detection=False, silent=True)
            if isinstance(dfs, list) and len(dfs) > 0:
                for df in dfs:
                    if not df.empty:
                        matched_path = df.iloc[0]["identity"]
                        person_name = os.path.basename(os.path.dirname(matched_path))
                        detected_categories.add(person_name)
        except Exception as e:
            pass # Face recognition failed or no face found
            
    # 2. Object Detection using YOLO
    try:
        results = yolo_model(file_path, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = yolo_model.names[cls_id]
                
                # If DeepFace already detected a specific person, we can optionally ignore the generic 'person' tag
                if class_name == 'person' and any(cat != 'person' for cat in detected_categories):
                    continue
                    
                detected_categories.add(class_name)
    except Exception as e:
        pass
        
    if not detected_categories:
        print("No faces or objects detected. Categorizing as 'other'.")
        detected_categories.add("other")
        
    return list(detected_categories)

def apply_categories(file_path: str, categories: list):
    success_paths = []
    file_name = os.path.basename(file_path)
    
    for category in categories:
        target_dir = os.path.join(CATEGORIZED_DIR, category.strip())
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, file_name)
        
        if os.path.exists(target_path):
            success_paths.append(target_path)
            continue
            
        try:
            os.link(file_path, target_path)
            success_paths.append(target_path)
        except OSError as e:
            if e.errno == 18: # Cross-device link error
                try:
                    os.symlink(os.path.abspath(file_path), target_path)
                    success_paths.append(target_path)
                except Exception:
                    shutil.copy2(file_path, target_path)
                    success_paths.append(target_path)
            else:
                print(f"Error creating link for: {target_path} -> {e}")
                
    return success_paths

def process_and_categorize(file_path: str):
    categories = predict_categories(file_path)
    if not categories:
        return [], []
        
    success_paths = apply_categories(file_path, categories)
    
    print(f"Detected categories: {', '.join(categories)}")
    print(f"Successfully categorized into: {', '.join(success_paths)}")
    print("-" * 50)
    
    return categories, success_paths

def main():
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(CATEGORIZED_DIR, exist_ok=True)
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    
    # Supported image formats
    valid_extensions = {".jpg", ".jpeg", ".png"}
    
    # If a single file name is given from command line, process only that file
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
        process_and_categorize(target_path)
    else:
        # Process all images in the test_images folder
        files_to_process = []
        if os.path.exists(TEST_IMAGES_DIR):
            for file_name in os.listdir(TEST_IMAGES_DIR):
                ext = os.path.splitext(file_name)[1].lower()
                if ext in valid_extensions:
                    files_to_process.append(os.path.join(TEST_IMAGES_DIR, file_name))
        
        if not files_to_process:
            print(f"Warning: No images found to process in '{TEST_IMAGES_DIR}' directory.")
            print(f"Please place photos to be analyzed into the '{TEST_IMAGES_DIR}' folder and run again.")
            return
            
        print(f"Found a total of {len(files_to_process)} photos. Starting process...\n")
        for idx, file_path in enumerate(files_to_process, 1):
            print(f"--- [{idx}/{len(files_to_process)}] ---")
            process_and_categorize(file_path)

if __name__ == "__main__":
    main()
