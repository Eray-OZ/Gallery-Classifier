import gradio as gr
import os
import shutil
import pandas as pd
from vision_os import predict_categories, apply_categories, TEST_IMAGES_DIR

def analyze_images(files):
    if not files:
        return pd.DataFrame(columns=["File Name", "Categories"]), []
    
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    
    data = []
    gallery_images = []
    
    for file_path in files:
        original_filename = os.path.basename(file_path)
        target_path = os.path.join(TEST_IMAGES_DIR, original_filename)
        
        try:
            shutil.copy2(file_path, target_path)
        except Exception as e:
            continue
            
        cats = predict_categories(target_path)
        cat_str = ", ".join(cats) if cats else ""
        
        data.append({
            "File Name": original_filename,
            "Categories": cat_str
        })
        # Gradio gallery supports tuples of (image_path, caption)
        gallery_images.append((target_path, original_filename))
        
    df = pd.DataFrame(data)
    return df, gallery_images

def save_and_categorize(df):
    # Depending on Gradio version, df might be passed as a dict, list, or DataFrame
    if isinstance(df, dict):
        df = pd.DataFrame(df)
    elif isinstance(df, list):
        df = pd.DataFrame(df, columns=["File Name", "Categories"])
        
    if df is None or df.empty:
        return "No data to process. Please upload and analyze photos first."
        
    results = []
    for _, row in df.iterrows():
        file_name = row["File Name"]
        cat_str = str(row["Categories"])
        
        file_path = os.path.join(TEST_IMAGES_DIR, file_name)
        
        cats = [c.strip() for c in cat_str.split(",") if c.strip()]
        
        if not cats:
            results.append(f"[SKIPPED] {file_name}: No categories specified.")
            continue
            
        if not os.path.exists(file_path):
             results.append(f"[ERROR] {file_name}: File not found in test_images.")
             continue
             
        paths = apply_categories(file_path, cats)
        results.append(f"[SUCCESS] {file_name}: Categorized into {', '.join(cats)}.")
        
    return "\n\n".join(results)

with gr.Blocks(title="Photo Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Photo Classifier (Interactive Mode)")
    gr.Markdown("Upload your photos below. The system will predict categories. You can review and edit these categories in the table before saving.")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload / Select Photos", file_count="multiple", file_types=["image"], type="filepath")
            analyze_btn = gr.Button("1. Analyze & Preview", variant="secondary")
            
        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Uploaded Images Preview", show_label=True, columns=4, height=300)
            
    gr.Markdown("### 2. Edit Categories")
    gr.Markdown("Click on the 'Categories' column below to change or add new categories (comma-separated). For example, change 'other' to 'vacation, family' to put the photo in two folders.")
    
    # Interactive dataframe so user can edit categories
    category_df = gr.Dataframe(
        headers=["File Name", "Categories"],
        datatype=["str", "str"],
        col_count=(2, "fixed"),
        interactive=True,
        wrap=True
    )
    
    save_btn = gr.Button("3. Save & Categorize", variant="primary")
    output_text = gr.Textbox(label="Final Results", lines=10, interactive=False)
    
    analyze_btn.click(fn=analyze_images, inputs=file_input, outputs=[category_df, gallery])
    save_btn.click(fn=save_and_categorize, inputs=category_df, outputs=output_text)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
