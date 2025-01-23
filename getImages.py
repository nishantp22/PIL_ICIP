import requests
from PIL import Image
from io import BytesIO

image_url="https://drive.google.com/file/d/1V-l3ER1ugcLoZaR-POaTW_IxdScjkoJA/view"

response = requests.get(image_url)
img = Image.open(BytesIO(response.content))

desired_size = (560, 848) 
# img_resized = img.resize(desired_size)

img.save('fgp.jpg')

