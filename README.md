# PIL_ICIP

Run `pip install -r requirements.txt` to install dependencies.

To download MODnet, run

`gdown --id 1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz -O pretrained/modnet_photographic_portrait_matting.ckpt`

Run `python script.py` to generate an image from Flux-Controlnet with a Canny image.

Run `python script2.py` to remove foreground from generated image, and apply ICLight to the original foreground with the new background.

Run `python script3.py` to apply HSV chanages.

