# open image and change the size to 1024 1024 

from PIL import Image

image = Image.open("/home/paulj/projects/collosal/4D-Humans/data/smpl_uv_20200910/texture.png")
image = image.resize((512,512))
image.save("/home/paulj/projects/collosal/4D-Humans/data/smpl_uv_20200910/texture_red.png")