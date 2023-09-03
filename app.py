import streamlit as st
from PIL import Image
from diffusers import DiffusionPipeline
import torch
import gc  # Import the garbage collector
import base64

def generate_image(prompt_text):
    # Initialize the model within the function
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

    with st.spinner("Generating image..."):
        # Move the pipeline to GPU
        pipe.to("cuda")

        # If using torch >= 2.0, improve inference speed with torch.compile
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        # Generate the image
        with torch.no_grad():
            image = pipe(prompt=prompt_text).images[0]
            image_path = "generated_image.png"
            image.save(image_path)
            del image

        # Move the model back to CPU to free up GPU memory
        pipe.to("cpu")
        del pipe  # Delete the model
        gc.collect()  # Collect garbage
        flush_gpu_memory()

    return image_path

def flush_gpu_memory():
    torch.cuda.empty_cache()

def convert_to_png(image_path):
    """
    Convert image to PNG format.
    """
    with Image.open(image_path) as img:
        png_path = "converted_image.png"
        img.save(png_path, "PNG")
    return png_path

def main():
    # Sidebar for user input
    st.title("Image Generation from Text")
    prompt_text = st.text_input("Enter your prompt:", "A detailed oil painting about an astronaut riding a Yu-Gi-Oh's Blue-Eyes White Dragon")
    generate_button = st.button("Generate Image")

    if generate_button:
        # Check if the image is already generated for the same prompt
        if 'last_prompt' in st.session_state and st.session_state.last_prompt == prompt_text:
            image_path = st.session_state.image_path
        else:
            image_path = generate_image(prompt_text)
            st.session_state.last_prompt = prompt_text
            st.session_state.image_path = image_path

        png_path = convert_to_png(image_path)
        st.image(png_path, caption="Generated Image", use_column_width=True)

if __name__ == '__main__':
    main()
