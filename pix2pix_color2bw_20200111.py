import os
from glob import glob
from PIL import Image

dir_color_images = "C:/Users/PARKJaehee/Desktop/jeje50/"
dir_dst = "C:/Users/PARKJaehee/Desktop/jeje50/BW/"
os.makedirs(dir_dst, exist_ok=True)

list_path_color_images = glob(os.path.join(dir_color_images, "*.jpg"))  # glod 는 파일들의 경로를 가져옴
print(len(list_path_color_images))
list_path_color_images.sort()

count = 0
for path in list_path_color_images:
    image = Image.open(path).convert("L")
    image.save(os.path.join(dir_dst, os.path.basename(path)))# os.path.join 을 하면 앞의 인수와 뒤의 인수의 경로를 합쳐줌
    # image.show()
    # if count == 1:
    #    break
    count = count + 1
