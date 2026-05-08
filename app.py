import gradio as gr
import os
import shutil
from vision_os import process_and_categorize, TEST_IMAGES_DIR

def process_images(files):
    if not files:
        return "Please select at least one photo."
    
    results = []
    
    # Ensure test_images directory exists
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    
    for file_path in files:
        original_filename = os.path.basename(file_path)
        target_path = os.path.join(TEST_IMAGES_DIR, original_filename)
        
        try:
            shutil.copy2(file_path, target_path)
        except Exception as e:
            results.append(f"[ERROR] {original_filename}: Failed to copy file. ({e})")
            continue
            
        cats, paths = process_and_categorize(target_path)
        
        if not cats:
            results.append(f"[SKIPPED] {original_filename}: No categories or faces detected.")
        else:
            cat_str = ", ".join(cats)
            results.append(f"[SUCCESS] {original_filename}: Categorized as {cat_str}.")
            
    return "\n\n".join(results)

with gr.Blocks(title="Photo Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Photo Classifier")
    gr.Markdown("Upload your photos below. The system will analyze faces (DeepFace) and objects (YOLO) and automatically categorize them into the respective folders.")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload / Select Photos", file_count="multiple", file_types=["image"], type="filepath")
            submit_btn = gr.Button("Classify", variant="primary")
            
        with gr.Column():
            output_text = gr.Textbox(label="Results", lines=15, interactive=False)
            
    submit_btn.click(fn=process_images, inputs=file_input, outputs=output_text)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
