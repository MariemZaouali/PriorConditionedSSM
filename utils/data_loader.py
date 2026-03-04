import torch
import torch.utils.data as data
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import os
import random
import torchvision.transforms as transforms


def cv_random_flip(img_A, img_B, label):
    flag = random.randint(0, 1)
    if flag == 1:
        img_A = img_A.transpose(Image.FLIP_LEFT_RIGHT)
        img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img_A, img_B, label


def randomCrop_Mosaic(imageA, imageB, labelmap, cropsize):
    channels = len(imageA.shape)
    if channels == 3:
        h, w, c = imageA.shape
    else:
        h, w = imageA.shape

    if (w > cropsize[0]):  # w
        margins = w - cropsize[0]
        margins_left = int(margins / 2)
        margins_right = cropsize[0] + margins_left
        imageA = imageA[:, margins_left:margins_right, :]
        imageB = imageB[:, margins_left:margins_right, :]
        labelmap = labelmap[:, margins_left:margins_right]
    if (h > cropsize[1]):  # h
        margins = h - cropsize[1]
        margins_top = int(margins / 2)
        margins_bottom = cropsize[1] + margins_top
        imageA = imageA[margins_top:margins_bottom, :, :]
        imageB = imageB[margins_top:margins_bottom, :, :]
        labelmap = labelmap[margins_top:margins_bottom, :]

    return imageA, imageB, labelmap


def randomCrop(image_A, image_B, label):
    """
    Asymmetric Crops , Lots of image that has salient object not in the center
    """
    border = 30
    image_A = np.pad(image_A, ((border, border), (border, border), (0, 0)), 'reflect')
    image_B = np.pad(image_B, ((border, border), (border, border), (0, 0)), 'reflect')
    label = np.pad(label, ((border, border), (border, border)), 'reflect')
    H = image_A.shape[0]
    W = image_A.shape[1]
    ind_H = random.randint(0, H - 256)
    ind_W = random.randint(0, W - 256)

    image_A = image_A[ind_H:ind_H + 256, ind_W:ind_W + 256, :]
    image_B = image_B[ind_H:ind_H + 256, ind_W:ind_W + 256, :]
    label = label[ind_H:ind_H + 256, ind_W:ind_W + 256]

    return image_A, image_B, label


def randomRotation(image_A, image_B, label):
    """
     同时对image和label进行随机旋转
    """
    mode = random.randint(0, 20)
    if mode == 1:
        (h, w) = image_A.shape[:2]
        center = (w / 2, h / 2)
        angle = random.randint(-15, 15)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        image_A = cv2.warpAffine(image_A, M, (w, h))
        image_B = cv2.warpAffine(image_B, M, (w, h))
        label = cv2.warpAffine(label.astype(np.float32), M, (w, h), borderValue=0)

    return image_A, image_B, label


def colorEnhance(image_A, image_B):
    bright_intensity = random.randint(5, 15) / 10.0
    image_A = ImageEnhance.Brightness(image_A).enhance(bright_intensity)
    image_B = ImageEnhance.Brightness(image_B).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image_A = ImageEnhance.Contrast(image_A).enhance(contrast_intensity)
    image_B = ImageEnhance.Contrast(image_B).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image_A = ImageEnhance.Color(image_A).enhance(color_intensity)
    image_B = ImageEnhance.Color(image_B).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image_A = ImageEnhance.Sharpness(image_A).enhance(sharp_intensity)
    image_B = ImageEnhance.Sharpness(image_B).enhance(sharp_intensity)

    return image_A, image_B


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(image):
        """
        增加高斯噪声
        """
        img = np.asarray(image) / 255.0
        noise = np.random.normal(mean, sigma, img.shape)
        gaussianNoisy = img + noise
        gaussianNoisy = np.uint8(np.clip(gaussianNoisy * 255, 0, 255))
        return Image.fromarray(gaussianNoisy)

    return gaussianNoisy(image)


def randomPeper(img):
    """
    随机添加椒盐噪声
    """
    img = np.asarray(img) / 255.0
    noise = np.random.random(img.shape)
    img[noise > 0.9985] = 1
    img[noise < 0.0015] = 0

    return Image.fromarray(np.uint8(img * 255))


class ChangeDataset(data.Dataset):
    def __init__(self, img_path, gt_path, mosaic_ratio=0.25, transform_med=True):
        super(ChangeDataset, self).__init__()

        img_root_path = img_path
        gt_root_path = gt_path
        self.mosaic_ratio = mosaic_ratio
        self.transform_med = transform_med

        self.img_A_path_list = os.path.join(img_root_path, 'A')
        self.img_B_path_list = os.path.join(img_root_path, 'B')
        self.gt_path_list = os.path.join(gt_root_path, 'label')

        # Check if directories exist and have files
        if not os.path.exists(self.gt_path_list):
            raise ValueError(f"Label directory not found: {self.gt_path_list}")
        if not os.path.exists(self.img_A_path_list):
            raise ValueError(f"Image A directory not found: {self.img_A_path_list}")
        if not os.path.exists(self.img_B_path_list):
            raise ValueError(f"Image B directory not found: {self.img_B_path_list}")

        file_name_list = os.listdir(self.gt_path_list)
        if len(file_name_list) == 0:
            raise ValueError(f"No files found in label directory: {self.gt_path_list}")
        
        self.file_name_list = [x.split('.')[0] for x in file_name_list]
        print(f"Found {len(self.file_name_list)} training samples")

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):

        file_name = self.file_name_list[index]
        img_A, img_B, label = self.load_img_and_mask(file_name)

        if self.transform_med:
            seed = random.randint(0, 2022)
            random.seed(seed)

            if random.random() < self.mosaic_ratio:
                img_A, img_B, label = self.load_mosaic_img_and_mask(file_name)

        img_A = np.transpose(img_A, (2, 0, 1))
        img_B = np.transpose(img_B, (2, 0, 1))

        img_A_ts = torch.from_numpy(img_A).float()
        img_B_ts = torch.from_numpy(img_B).float()
        label_ts = torch.from_numpy(label).long()

        return img_A_ts, img_B_ts, label_ts

    def load_img_and_mask(self, file_name):
        img_A = Image.open(os.path.join(self.img_A_path_list, file_name + '.tif'))
        img_B = Image.open(os.path.join(self.img_B_path_list, file_name + '.tif'))
        label = Image.open(os.path.join(self.gt_path_list, file_name + '.tif'))

        img_A, img_B, label = cv_random_flip(img_A, img_B, label)
        img_A, img_B, label = randomCrop(np.array(img_A), np.array(img_B), np.array(label))

        img_A, img_B = colorEnhance(Image.fromarray(img_A), Image.fromarray(img_B))
        img_A = randomGaussian(img_A)
        img_B = randomGaussian(img_B)
        img_A = randomPeper(img_A)
        img_B = randomPeper(img_B)

        img_A = np.array(img_A)
        img_B = np.array(img_B)

        return img_A, img_B, label

    def load_mosaic_img_and_mask(self, file_name):
        s = 256
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in [s / 2, s / 2]]  # mosaic center x, y
        indices = [random.randint(0, len(self.file_name_list) - 1) for _ in range(3)]

        # 3个额外图像索引
        file_name_list = [file_name] + [self.file_name_list[i] for i in indices]

        img_A_list = []
        img_B_list = []
        label_list = []

        for index, f_name in enumerate(file_name_list):

            img_A = Image.open(os.path.join(self.img_A_path_list, f_name + '.tif'))
            img_B = Image.open(os.path.join(self.img_B_path_list, f_name + '.tif'))
            label = Image.open(os.path.join(self.gt_path_list, f_name + '.tif'))

            img_A, img_B, label = cv_random_flip(img_A, img_B, label)

            img_A = Image.fromarray(np.array(img_A))
            img_B = Image.fromarray(np.array(img_B))

            img_A = np.array(img_A)
            img_B = np.array(img_B)
            label = np.array(label)

            if index == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - s, 0), max(yc - s, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = s - (x2a - x1a), s - (y2a - y1a), s, s  # xmin, ymin, xmax, ymax (small image)
            elif index == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - s, 0), min(xc + s, s * 2), yc
                x1b, y1b, x2b, y2b = 0, s - (y2a - y1a), min(s, x2a - x1a), s
            elif index == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - s, 0), yc, xc, min(s * 2, yc + s)
                x1b, y1b, x2b, y2b = s - (x2a - x1a), 0, s, min(y2a - y1a, s)
            elif index == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + s, s * 2), min(s * 2, yc + s)
                x1b, y1b, x2b, y2b = 0, 0, min(s, x2a - x1a), min(s, y2a - y1a)

            # Extract patches with proper bounds checking
            patch_A = img_A[y1b:y2b, x1b:x2b, :]
            patch_B = img_B[y1b:y2b, x1b:x2b, :]
            patch_label = label[y1b:y2b, x1b:x2b]
            
            img_A_list.append(patch_A)
            img_B_list.append(patch_B)
            label_list.append(patch_label)

        img_A = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)  # base image with 4 tiles
        img_B = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        label = np.full((s * 2, s * 2), 0, dtype=np.uint8)

        for i, index in enumerate([0, 1, 2, 3]):

            if index == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - s, 0), max(yc - s, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = s - (x2a - x1a), s - (y2a - y1a), s, s  # xmin, ymin, xmax, ymax (small image)
            elif index == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - s, 0), min(xc + s, s * 2), yc
                x1b, y1b, x2b, y2b = 0, s - (y2a - y1a), min(s, x2a - x1a), s
            elif index == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - s, 0), yc, xc, min(s * 2, yc + s)
                x1b, y1b, x2b, y2b = s - (x2a - x1a), 0, s, min(y2a - y1a, s)
            elif index == 3:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + s, s * 2), min(s * 2, yc + s)
                x1b, y1b, x2b, y2b = 0, 0, min(s, x2a - x1a), min(s, y2a - y1a)

            # Get the patch and its actual size
            patch_A = img_A_list[i]
            patch_B = img_B_list[i]
            patch_label = label_list[i]
            
            # Calculate the actual patch size
            patch_h, patch_w = patch_A.shape[:2]
            
            # Calculate the target region size
            target_h = y2a - y1a
            target_w = x2a - x1a
            
            # If patch size doesn't match target size, resize the patch
            if patch_h != target_h or patch_w != target_w:
                # Resize patch to match target region size
                patch_A = np.array(Image.fromarray(patch_A).resize((target_w, target_h), Image.BILINEAR))
                patch_B = np.array(Image.fromarray(patch_B).resize((target_w, target_h), Image.BILINEAR))
                patch_label = np.array(Image.fromarray(patch_label).resize((target_w, target_h), Image.NEAREST))

            # Now assign the resized patch to the target region
            img_A[y1a:y2a, x1a:x2a] = patch_A
            img_B[y1a:y2a, x1a:x2a] = patch_B
            label[y1a:y2a, x1a:x2a] = patch_label

        # Call randomCrop_Mosaic once and unpack all three outputs
        img_A, img_B, label = randomCrop_Mosaic(img_A, img_B, label, (256, 256))
        
        # Ensure all outputs are exactly 256x256
        # Convert to PIL Image for resizing if needed
        if img_A.shape[:2] != (256, 256):
            img_A = Image.fromarray(img_A).resize((256, 256), Image.BILINEAR)
            img_A = np.array(img_A)
        if img_B.shape[:2] != (256, 256):
            img_B = Image.fromarray(img_B).resize((256, 256), Image.BILINEAR)
            img_B = np.array(img_B)
        if label.shape[:2] != (256, 256):
            label = Image.fromarray(label).resize((256, 256), Image.NEAREST)
            label = np.array(label)

        return img_A, img_B, label


class Test_ChangeDataset(data.Dataset):
    def __init__(self, img_path, gt_path, transform_med=True):
        super(Test_ChangeDataset, self).__init__()

        img_root_path = img_path
        gt_root_path = gt_path
        self.transform_med = transform_med

        self.img_A_path_list = os.path.join(img_root_path, 'A')
        self.img_B_path_list = os.path.join(img_root_path, 'B')
        self.gt_path_list = os.path.join(gt_root_path, 'label')

        file_name_list = os.listdir(self.gt_path_list)
        self.file_name_list = [x.split('.')[0] for x in file_name_list]

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):

        file_name = self.file_name_list[index]

        img_A = Image.open(os.path.join(self.img_A_path_list, file_name + '.tif'))
        img_B = Image.open(os.path.join(self.img_B_path_list, file_name + '.tif'))
        label = Image.open(os.path.join(self.gt_path_list, file_name + '.tif'))

        img_A = np.array(img_A)
        img_B = np.array(img_B)
        label = np.array(label)

        img_A = np.transpose(img_A, (2, 0, 1))
        img_B = np.transpose(img_B, (2, 0, 1))

        img_A_ts = torch.from_numpy(img_A).float()
        img_B_ts = torch.from_numpy(img_B).float()
        label_ts = torch.from_numpy(label).long()

        return img_A_ts, img_B_ts, label_ts, file_name


def get_loader(img_path, gt_path, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, mosaic_ratio=0.25):
    dataset = ChangeDataset(img_path, gt_path, mosaic_ratio=mosaic_ratio, transform_med=True)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


def get_test_loader(img_path, gt_path, batch_size=32, shuffle=False, num_workers=4, pin_memory=True):
    dataset = Test_ChangeDataset(img_path, gt_path, transform_med=False)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader
