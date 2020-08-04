from PIL import Image, ImageDraw
from math import floor

Image.MAX_IMAGE_PIXELS = 933120000

def resize(file,x_size,y_size):
    # Load image:
    input_image = Image.open(file)
    input_pixels = input_image.load()

    new_size = (x_size, y_size)

    # Create output image
    output_image = Image.new("RGB", new_size)
    draw = ImageDraw.Draw(output_image)
    x_scale = input_image.width / output_image.width
    y_scale = input_image.height / output_image.height

    # Copy pixels
    for x in range(output_image.width):
        for y in range(output_image.height):
            xp, yp = floor(x * x_scale), floor(y * y_scale)
            draw.point((x, y), input_pixels[xp, yp])

    #output_image.save("scaled.png")
    return output_image
