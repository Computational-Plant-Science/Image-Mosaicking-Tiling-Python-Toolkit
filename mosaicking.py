import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def tile_imageRGB(input_path, tile_size=(256, 256), overlap=0):

    # Open the input image
    input_image = Image.open(input_path)

    # Convert image to numpy array
    img_array = np.array(input_image)
    # img_height, img_width, channels = img_array.shape
    # Get image dimensions
    gray=0
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=-1)
        img_height, img_width, channels = img_array.shape
        gray=1
    else:
        img_height, img_width, channels = img_array.shape
    # print(img_array.shape)

    # Calculate the number of tiles in each dimension
    num_tiles_x = (img_width - overlap) // (tile_size[0] - overlap)
    num_tiles_y = (img_height - overlap) // (tile_size[1] - overlap)
    tiles = []
    # Extract tiles
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            # Calculate the starting position of each tile
            start_w = i * (tile_size[0] - overlap)
            start_h = j * (tile_size[1] - overlap)

            # Calculate the ending position of each tile
            end_w = start_w + tile_size[0]
            end_h = start_h + tile_size[1]

            # Crop the tile
            tile = img_array[start_h:end_h, start_w:end_w, :]
            tiles.append(tile)
    return tiles,gray,img_height, img_width


def mosaicking_mage(tiles, tile_size, overlap, img_height, img_width, gray=0, val=50):
    # Create an empty array to store the reconstructed image
    reconstructed_image = np.zeros((img_height, img_width, 3), dtype=np.uint64)
    overlap_count = np.zeros((img_height, img_width, 3), dtype=np.uint64)

    # Counter to keep track of the current tile index
    tile_index = 0

    for i in range(img_width // (tile_size[0] - overlap)):
        for j in range(img_height // (tile_size[1] - overlap)):
            # Calculate the starting position of each tile
            start_w = i * (tile_size[0] - overlap)
            start_h = j * (tile_size[1] - overlap)

            # Calculate the ending position of each tile
            end_w = min(start_w + tile_size[0], img_width)
            end_h = min(start_h + tile_size[1], img_height)

            # Check if the tile index is within the range of the tiles list
            if tile_index < len(tiles):
                # Accumulate the tile values and count the number of times each pixel is added
                tile_region = tiles[tile_index][:end_h - start_h, :end_w - start_w, :]
                reconstructed_image[start_h:end_h, start_w:end_w, :] += tile_region.astype(np.uint64)
                overlap_count[start_h:end_h, start_w:end_w, :] += 1

                # Increment the tile index
                tile_index += 1

    # Normalize the overlapping regions by dividing with the accumulated counts
    reconstructed_image = np.divide(reconstructed_image, np.maximum(overlap_count, 1))

    # Clip values to the valid range for uint8
    reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)

    # Trim the additional pixels if necessary
    reconstructed_image = reconstructed_image[:img_height - val, :img_width, :]

    # Convert back to grayscale if needed
    if gray:
        reconstructed_image = np.squeeze(reconstructed_image, axis=-1)

    # Convert to Image format
    reconstructed_image = Image.fromarray(reconstructed_image)

    return reconstructed_image


image_path = 'test_image.jpg'

tile_size = (200, 200)
overlap=0
tiles, gray,img_height, img_width = tile_imageRGB(image_path, tile_size, overlap)
Image.fromarray(tiles[0])
plt.imshow(Image.fromarray(tiles[0]),aspect="auto",cmap='gray')
print("Done")

reconstructed_image = mosaicking_mage(tiles, tile_size, overlap, img_height=img_height, img_width=img_width, gray=0,val=0)
plt.imshow(reconstructed_image, cmap='gray')
print("Done")
