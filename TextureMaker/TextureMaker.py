import numpy as np
from PIL import Image
import os

def generate_base_color_map(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGBA')  # Convert to RGBA
    base, ext = os.path.splitext(image_path)
    base_color_map_path = f"{base}_bc.png"
    image.save(base_color_map_path)
    print(f"BaseColor map saved to: {base_color_map_path}")
    
def generate_normal_map(image_path):
    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    width, height = image.size
    image_array = np.asarray(image, dtype=np.float32)

    # Initialize the normal map array
    normal_map = np.zeros((height, width, 3), dtype=np.float32)

    # Sobel filters for x and y gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    # Compute gradients
    gradient_x = np.zeros_like(image_array)
    gradient_y = np.zeros_like(image_array)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = image_array[i-1:i+2, j-1:j+2]
            gradient_x[i, j] = np.sum(sobel_x * region)
            gradient_y[i, j] = np.sum(sobel_y * region)

    # Normalize the gradients
    max_gradient = max(np.max(gradient_x), np.max(gradient_y))
    gradient_x /= max_gradient
    gradient_y /= max_gradient

    # Generate normal map
    normal_map[:, :, 0] = gradient_x  # X
    normal_map[:, :, 1] = gradient_y  # Y
    normal_map[:, :, 2] = 1.0  # Z component

    # Normalize to range [0, 1]
    normal_map = (normal_map + 1.0) / 2.0

    # Convert to 8-bit and save
    normal_map_image = Image.fromarray((normal_map * 255).astype(np.uint8))

    # Save the normal map with the new filename
    base, ext = os.path.splitext(image_path)
    normal_map_path = f"{base}_nm.normal.png"
    normal_map_image.save(normal_map_path)

    print(f"Normal map saved to: {normal_map_path}")

def generate_opacity_data_map(image_path):
    # Load the image and check for alpha channel
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        alpha_channel = image.split()[3]
        opacity_data_map = Image.new('L', image.size)
        opacity_data_map.paste(alpha_channel, None)
        base, ext = os.path.splitext(image_path)
        opacity_data_map_path = f"{base}_op.data.png"
        opacity_data_map.save(opacity_data_map_path)
        print(f"Opacity Data map saved to: {opacity_data_map_path}")
    else:
        print("No alpha channel found in the image.")
        
def generate_roughness_data_map(image_path):
    # Placeholder function for roughness map generation
    # This can be customized based on the desired roughness representation
    # For demonstration, let's create a roughness map based on the grayscale image intensity

    # Load the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    width, height = image.size
    image_array = np.asarray(image, dtype=np.float32)

    # Normalize intensity values to [0, 1]
    normalized_intensity = image_array / 255.0

    # Invert intensity to get roughness (simply for demonstration)
    roughness_map = 1.0 - normalized_intensity

    # Convert to 8-bit and save
    roughness_data_map = Image.fromarray((roughness_map * 255).astype(np.uint8))

    # Save the roughness data map with the new filename
    base, ext = os.path.splitext(image_path)
    roughness_data_map_path = f"{base}_r.data.png"
    roughness_data_map.save(roughness_data_map_path)
    print(f"Roughness Data map saved to: {roughness_data_map_path}")
    
if __name__ == "__main__":
    image_path = input("Please enter the path to the image: ")
    if os.path.isfile(image_path):
        generate_base_color_map(image_path)
        generate_normal_map(image_path)
        generate_opacity_data_map(image_path)
        generate_roughness_data_map(image_path)
    else:
        print("The provided path does not exist or is not a file.")
