from utils.bgrem import apply_remove_background
from utils.lama import apply_big_lama
from utils.ICLight import apply_relight
from utils.modnetRun import get_matte_from_modnet
import warnings
warnings.filterwarnings("ignore")



apply_remove_background(image_path="images/original_image.png", outName="original_BGRemoved", save_dir="images", threshold=0.5)

apply_remove_background(image_path="images/controlnet_output.png",outName="controlnet_BGRemoved", save_dir="images", threshold=0.5)

apply_big_lama(input_image="images/controlnet_BGRemoved.png",mask="images/controlnet_BGRemovedMask.png")

apply_relight(input_fg="images/original_BGRemoved.png",input_bg='images/new_background.png')

get_matte_from_modnet(
    image_path='images/RelightOutput_image_1.png', 
    output_path='images/image_matte.png', 
    ckpt_path='models/modnet_photographic_portrait_matting.ckpt',
    refinement_method='sharpen',  #use 'threshold', 'sharpen', or 'blur'
    threshold=128,  #only needed if method is 'threshold'
    blur_radius=1   #only needed if method is 'blur'
)


