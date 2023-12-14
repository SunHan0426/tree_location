import colorsys
import copy
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from nets.pspnet import PSPNet as pspnet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config
import matplotlib.pyplot as plt
import os
import logging
import sys

class PSPNet(object):
    _defaults = {
        "model_path": 'model_data/improved_net.pth',
        "num_classes": 2,
        "backbone": "mobilenet",
        "input_shape": [473, 473],
        "downsample_factor": 8,
        "mix_type": 0,
        "cuda": True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()
        
        show_config(**self._defaults)

    def generate(self, onnx=False):

        self.net = pspnet(num_classes=self.num_classes, downsample_factor=self.downsample_factor, pretrained=False,
                          backbone=self.backbone, aux_branch=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def detect_image(self, image, count=False, name_classes=None):
        image = cvtColor(image)
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

            with open('id.txt', 'r') as file:
                content = file.read()
            try:
                id_predict = int(content.split('=')[1])
            except ValueError:
                print('Failed to read id from the file.')

            log_file = f'logs/log_{id_predict:0>3d}.txt'
            logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w')

            class LoggerWriter:
                def __init__(self):
                    self.terminal = sys.stdout

                def write(self, message):
                    logging.info(message)
                    sys.__stdout__.write(message)

                def flush(self):
                    pass

            sys.stdout = LoggerWriter()

            counts_trunk = 0
            depth_sum = 0
            print("pr的shape: ", pr.shape)
            npy = f'npy/distance_npy_ZED_{id_predict:0>3d}.npy'
            depth_array = np.load(npy)
            print(f"LOAD npy_ZED_{id_predict:0>3d} SUCCESS.")
            print("depth_array(.npy) shape：", depth_array.shape)
            print(depth_array)
            pixels_trunk = np.zeros_like(depth_array)
            for i in range(0, pr.shape[0]):
                for j in range(0, pr.shape[1]):
                    if pr[i, j] == 1:
                        counts_trunk += 1
                        if np.isnan(depth_array[i, j]):
                            depth_array[i, j] = 0
                        depth_sum += depth_array[i, j]
                        depth_sum = round(depth_sum, 4)
                        if not np.isnan(depth_array[i, j]):
                            pixels_trunk[i, j] = depth_array[i, j]

            print("Shape before eliminate 0", pixels_trunk.shape)
            pixels_trunk_nozero = pixels_trunk[pixels_trunk != 0]
            print("Shape after eliminate 0", pixels_trunk_nozero.shape)
            Savepath_nptrunk = os.path.join("./npy", "pixels_trunk_{:0>3d}.npy".format(id_predict))
            Savepath_npnozero = os.path.join("./npy", "pixels_trunk_nozero_{:0>3d}.npy".format(id_predict))
            np.save(Savepath_nptrunk, pixels_trunk)
            print("{:0>3d} trunk with zero npy SAVED.".format(id_predict))
            np.save(Savepath_npnozero, pixels_trunk_nozero)
            print("{:0>3d} trunk no zero npy SAVED.".format(id_predict))

            print("The maximum and minimum depth of the original data:", np.max(pixels_trunk_nozero), np.min(pixels_trunk_nozero))
            average_depth = round(depth_sum / (counts_trunk + 1e-5), 4)
            print("Original data depth sum:", depth_sum)
            print("Number of trunk pixels in original data:", counts_trunk)
            print("Average trunk depth of original data:", average_depth)

            trunk_median = np.median(pixels_trunk_nozero)
            trunk_mad = np.median(np.abs(pixels_trunk_nozero - trunk_median))
            z_scores = (pixels_trunk_nozero - trunk_median) / trunk_mad
            threshold_z = 2
            # The threshold value threshold_z had verified by the paper experiment
            pixels_trunk_z = pixels_trunk_nozero[abs(z_scores) < threshold_z]

            print("The maximum and minimum depth of the Z-Score data:", np.max(pixels_trunk_z), np.min(pixels_trunk_z))
            counts_trunk_z = len(pixels_trunk_z)
            depth_sum_z = round(np.sum(pixels_trunk_z), 4)
            print("Z-Score data depth sum:", depth_sum_z)
            print("Number of trunk pixels in Z-Score data:", counts_trunk_z)
            average_depth_z = round(depth_sum_z / (counts_trunk_z + 1e-5), 4)
            print("Average trunk depth of Z-Score data:", average_depth_z)

        x_range = (0, 8)
        bin_width = 0.05

        plt.hist(pixels_trunk_z, bins=np.arange(x_range[0], x_range[1] + bin_width, bin_width), range=x_range,
                 color='blue', alpha=0.7)
        plt.xlabel('Value')
        plt.ylabel('Times')
        plt.title('zscore_data')
        plt.xlim(x_range)
        plt.grid(True)
        plt.savefig("chart/pixels_trunk_z_{:0>3d}.png".format(id_predict))
        # plt.show()
        ymin, ymax = plt.ylim()
        plt.clf()

        plt.hist(pixels_trunk_nozero, bins=np.arange(x_range[0], x_range[1] + bin_width, bin_width), range=x_range,
                 color='blue', alpha=0.7)
        plt.xlabel('Value')
        plt.ylabel('Times')
        plt.title('origin_data')
        plt.xlim(x_range)
        plt.ylim(ymin, ymax)
        plt.grid(True)
        plt.savefig("chart/origin_data_{:0>3d}.png".format(id_predict))
        plt.clf()

        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|" % ("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|" % (str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
    
        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image = Image.fromarray(np.uint8(seg_img))
            image = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            image = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            image = Image.fromarray(np.uint8(seg_img))
        
        return image

    def get_FPS(self, image, test_interval):
        image = cvtColor(image)
        image_data, nw, nh = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
                pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')
        input_layer_names = ["images"]
        output_layer_names = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net, im, f =model_path, verbose =False, opset_version=12,
                        training = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding =True,
                        input_names =input_layer_names,
                        output_names =output_layer_names,
                        dynamic_axes =None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))
    
    def get_miou_png(self, image):
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image_data, nw, nh = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image
