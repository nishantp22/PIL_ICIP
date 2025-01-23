from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")


image_with_details = Image.open('images/RelightOutput_image_2.png')
image_with_color_lighting = Image.open('images/RelightOutput_image_1.png')

image_with_details = image_with_details.convert('RGBA')
image_with_color_lighting = image_with_color_lighting.convert('RGBA')

details_array = np.array(image_with_details)
color_array = np.array(image_with_color_lighting)

def rgb_to_hsv(rgb):
    r, g, b = rgb / 255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx - mn
    if df == 0:
        h = 0
    elif mx == r:
        h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / df) + 120) % 360
    else:
        h = (60 * ((r - g) / df) + 240) % 360
    v = mx
    s = 0 if mx == 0 else (df / mx)
    return h, s, v

def hsv_to_rgb(hsv):
    h, s, v = hsv
    c = v * s
    x = c * (1 - abs(((h / 60) % 2) - 1))
    m = v - c
    if 0 <= h < 60:
        r, g, b = c, x, 0
    elif 60 <= h < 120:
        r, g, b = x, c, 0
    elif 120 <= h < 180:
        r, g, b = 0, c, x
    elif 180 <= h < 240:
        r, g, b = 0, x, c
    elif 240 <= h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)

height, width, _ = details_array.shape
output_array = np.zeros_like(details_array)

for y in range(height):
    for x in range(width):
        alpha = details_array[y, x, 3]
        
        if alpha > 0:  
            h1, s1, v1 = rgb_to_hsv(details_array[y, x, :3]) 
            h2, s2, v2 = rgb_to_hsv(color_array[y, x, :3])    


            v = v1 * 0.45 + v2 * 0.55

            h,s=h2,s2

        # Slightly blend chrominance for smoother transitions

            output_array[y, x, :3] = hsv_to_rgb((h, s, v))
            output_array[y, x, 3] = alpha  
        else:
            output_array[y, x] = [0, 0, 0, 0]

final_image = Image.fromarray(output_array, 'RGBA')
final_image.save('images/relight_foreground_details.png')

foreground = Image.open("images/relight_foreground_details.png").convert("RGBA")  
background = Image.open("images/RelightOutput_image_3.png").convert("RGBA")

mask = Image.open('images/image_matte.png').convert('L') 

if foreground.size != background.size or foreground.size != mask.size:
    print("Foreground, background, or mask sizes do not match. Please ensure they have the same dimensions.")
    exit()

alpha = mask  

composed_image = Image.composite(foreground, background, alpha)

composed_image.save("final_image.png")
