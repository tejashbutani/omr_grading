import streamlit as st
import numpy as np
import json
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas
import base64
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_image_download_link(img, filename="image.png", text="Download image"):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def resize_image(image, max_width=1000):
    """Resize image while maintaining aspect ratio"""
    width, height = image.size
    if width > max_width:
        ratio = max_width / width
        new_width = max_width
        new_height = int(height * ratio)
        logger.debug(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

def convert_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

st.set_page_config(layout="wide")
st.title("🎯 OMR Bubble Mapper Tool")

# Initialize session state
if 'coords' not in st.session_state:
    st.session_state.coords = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = 1
if 'options_per_q' not in st.session_state:
    st.session_state.options_per_q = 4
if 'history' not in st.session_state:
    st.session_state.history = []
if 'history_index' not in st.session_state:
    st.session_state.history_index = -1
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'edit_question' not in st.session_state:
    st.session_state.edit_question = 1

def get_option_label(index):
    return chr(65 + index)  # A, B, C, D...

def save_state():
    if len(st.session_state.history) > st.session_state.history_index + 1:
        st.session_state.history = st.session_state.history[:st.session_state.history_index + 1]
    st.session_state.history.append(st.session_state.coords.copy())
    st.session_state.history_index = len(st.session_state.history) - 1

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    st.session_state.options_per_q = st.number_input(
        "Options per question", 
        min_value=2, 
        max_value=6, 
        value=st.session_state.options_per_q
    )
    
    drawing_mode = st.selectbox(
        "Drawing tool:",
        ("point", "circle", "transform"),
        index=0,
    )
    
    stroke_width = st.slider("Stroke width: ", 1, 25, 3)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⟲ Undo") and st.session_state.history_index > 0:
            st.session_state.history_index -= 1
            st.session_state.coords = st.session_state.history[st.session_state.history_index].copy()
            st.experimental_rerun()
    
    with col2:
        if st.button("⟳ Redo") and st.session_state.history_index < len(st.session_state.history) - 1:
            st.session_state.history_index += 1
            st.session_state.coords = st.session_state.history[st.session_state.history_index].copy()
            st.experimental_rerun()
    
    if st.button("🗑️ Clear Canvas"):
        st.session_state.coords = []
        st.session_state.current_question = 1
        st.session_state.history = []
        st.session_state.history_index = -1
        st.experimental_rerun()
    
    st.divider()
    st.session_state.edit_mode = st.checkbox("Edit Mode", value=st.session_state.edit_mode)
    if st.session_state.edit_mode:
        total_questions = len(st.session_state.coords) // st.session_state.options_per_q
        st.session_state.edit_question = st.number_input(
            "Edit Question Number",
            min_value=1,
            max_value=max(1, total_questions),
            value=st.session_state.edit_question
        )

uploaded_file = st.file_uploader("📤 Upload Cropped OMR Image", type=["jpg", "png"])

if uploaded_file:
    try:
        # Read image
        logger.debug("Reading uploaded image")
        image = Image.open(uploaded_file)
        logger.debug(f"Original image size: {image.size}")
        
        # Convert to RGB if needed
        img_array = np.array(image)
        if len(img_array.shape) == 2:
            logger.debug("Converting grayscale to RGB")
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(img_array)
        elif img_array.shape[2] == 4:
            logger.debug("Converting RGBA to RGB")
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            image = Image.fromarray(img_array)
        
        # Resize image
        resized_image = resize_image(image)
        logger.debug(f"Resized image size: {resized_image.size}")
        
        # Main content
        st.markdown("### Instructions")
        st.info("""
        1️⃣ Use the point tool to mark bubbles in order (A, B, C, D...)
        2️⃣ Points are automatically grouped into questions
        3️⃣ Use transform mode to adjust point positions if needed
        4️⃣ Enable Edit Mode to modify specific questions
        """)
        
        # Create columns for better layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            current_q = st.session_state.edit_question if st.session_state.edit_mode else st.session_state.current_question
            st.subheader(f"Marking Question {current_q}")
            
            # Show current options being marked
            options_marked = len(st.session_state.coords) % st.session_state.options_per_q
            options_text = " → ".join([
                f"[{'✓' if i < options_marked else ' '}] {get_option_label(i)}"
                for i in range(st.session_state.options_per_q)
            ])
            st.write(f"Options: {options_text}")
            
            # Create canvas
            logger.debug("Creating canvas with PIL Image")
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=stroke_width,
                stroke_color="rgba(255, 165, 0, 0.8)",
                background_image=resized_image,  # Pass PIL Image directly
                drawing_mode=drawing_mode,
                point_display_radius=3,
                key="canvas",
                width=resized_image.width,
                height=resized_image.height,
                update_streamlit=True,
                display_toolbar=True,
            )
        
        with col2:
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
                
                # Extract point coordinates
                current_coords = []
                for obj in objects:
                    if obj.get("type") in ["circle", "point"]:
                        x = int(obj.get("left", 0))
                        y = int(obj.get("top", 0))
                        # Scale coordinates if image was resized
                        if image.width > resized_image.width:
                            x = int(x * image.width / resized_image.width)
                            y = int(y * image.height / resized_image.height)
                        current_coords.append([x, y])
                
                # Update session state
                if current_coords != st.session_state.coords:
                    logger.debug(f"Updating coordinates: {len(current_coords)} points")
                    st.session_state.coords = current_coords
                    save_state()
                
                if current_coords:
                    total_questions = len(current_coords) // st.session_state.options_per_q
                    st.write(f"Total points marked: {len(current_coords)}")
                    st.write(f"Questions completed: {total_questions}")
                    
                    # Save buttons
                    if st.button("💾 Save Bubble Map"):
                        # Create organized format
                        organized = {}
                        for i in range(0, len(current_coords), st.session_state.options_per_q):
                            q_coords = current_coords[i:i + st.session_state.options_per_q]
                            if len(q_coords) == st.session_state.options_per_q:
                                q_num = f"question_{(i // st.session_state.options_per_q) + 1}"
                                organized[q_num] = {
                                    get_option_label(j): coord 
                                    for j, coord in enumerate(q_coords)
                                }
                        
                        # Save both formats
                        with open("bubble_map.json", "w") as f:
                            json.dump(current_coords, f)
                        with open("organized_bubble_map.json", "w") as f:
                            json.dump(organized, f)
                        st.success("Saved both raw and organized bubble maps!")
                    
                    # Show coordinates in a more organized way
                    st.markdown("### Marked Points")
                    for q in range(total_questions):
                        with st.expander(f"Question {q + 1}"):
                            q_coords = current_coords[q * st.session_state.options_per_q:(q + 1) * st.session_state.options_per_q]
                            for i, coord in enumerate(q_coords):
                                st.write(f"Option {get_option_label(i)}: {coord}")
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")
