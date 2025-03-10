import random
import math
import os
from PIL import Image
from tqdm import tqdm

def generate_multigrain_images(
    base_output_dir="Input0",
    num_dirs=10,
    count=500,
    size=128,
    num_grains=5,
    threshold=2.0
):
    """
    Generate a specified number of 128x128 'multi-grain' images, each with black boundaries
    on a white background, and save them in directories named Input0 to Input9.
    
    :param base_output_dir: Base directory name where images will be saved.
    :param num_dirs: Number of directories (Input0 to Input9).
    :param count: How many images to generate per directory.
    :param size: The size of each image (width=height).
    :param num_grains: Number of grains (seed points) to generate per image.
    :param threshold: Distance difference threshold for drawing boundaries.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for dir_index in range(num_dirs):
        output_dir = os.path.join(script_dir, f"{base_output_dir}{dir_index}")
        os.makedirs(output_dir, exist_ok=True)
        
        for i in tqdm(range(count), desc=f"Generating images for {output_dir}"):
            # Generate random seed points
            points = [(random.uniform(0, size), random.uniform(0, size)) for _ in range(num_grains)]
            
            # Create a blank (white) image
            image = Image.new("L", (size, size), 255)
            pixels = image.load()

            # Compute distances and determine boundaries
            for y in range(size):
                for x in range(size):
                    distances = [math.dist((x, y), p) for p in points]
                    distances.sort()
                    if (distances[1] - distances[0]) < threshold:
                        pixels[x, y] = 0   # black boundary
                    else:
                        pixels[x, y] = 255 # white

            # Save each image with a sequential name like "0.png", "1.png", ...
            filename = f"{i}.png"
            output_path = os.path.join(output_dir, filename)
            image.save(output_path)

if __name__ == "__main__":
    generate_multigrain_images()
