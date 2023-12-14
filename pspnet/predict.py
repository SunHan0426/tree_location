from PIL import Image
from pspnet import PSPNet
import os
from tqdm import tqdm


if __name__ == "__main__":

    pspnet = PSPNet()
    mode = "predict"
    count = True
    name_classes = ["background", "trunk"]

    with open('id.txt', 'r') as file:
        content = file.read()
    try:
        id_predict = int(content.split('=')[1])
    except ValueError:
        print('Failed to read id from the file.')

    if mode == "predict":
        i = 0
        while i == 0:
            img = f'img/RGB_LEFT_{id_predict:0>3d}.png'
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = pspnet.detect_image(image, count=count, name_classes=name_classes)
                r_image.save("./img_out/RGB_trunk_predicted_{:0>3d}.png".format(id_predict))
                # r_image.show()
            i += 1
        
    elif mode == "dir_predict":
        dir_origin_path = "img/"
        dir_save_path = "img_out/"
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path = os.path.join(dir_origin_path, img_name)
                image = Image.open(image_path)
                r_image = pspnet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
