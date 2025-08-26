import os
import cv2
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import tqdm
import h5pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Resize, InterpolationMode
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.degradation_utils import Degradation
from utils.image_utils import random_augmentation, crop_img, crop_img_inference

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class Inference(Dataset):
    def __init__(self, args):
        super(Inference, self).__init__()

        self.args = args
        self.toTensor = ToTensor()

        self.resize = Resize(size=(512, 512), interpolation=InterpolationMode.NEAREST)

        self._init_inference()

    def __getitem__(self, idx):
        lr_sample = self.lr[idx]

        lr, h, w = crop_img_inference(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)

        lr = self.toTensor(lr)

        return lr_sample["img"], lr, h, w

    def __len__(self):
        return len(self.lr)

    def _init_inference(self):
        inputs = self.args.data_file_dir
        # 支持多种图像格式
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(inputs + "/" + ext))
        self.lr = [{"img": x} for x in sorted(image_files)]

        # self.lr = [{"img": x} for x in sorted(glob.glob(inputs + "/*.png"))]

        print("Total inference images : {}".format(len(self.lr)))

class Validation(Dataset):
    def __init__(self, args):
        super(Validation, self).__init__()

        self.args = args
        self.toTensor = ToTensor()

        self.resize = Resize(size=(512, 512), interpolation=InterpolationMode.NEAREST)

        self._init_val()

    def __getitem__(self, idx):

        lr_sample = self.lr[idx]
        hr_sample = self.hr[idx]
        lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
        hr = crop_img(np.array(Image.open(hr_sample["img"]).convert('RGB')), base=16)

        lr = self.toTensor(lr)
        hr = self.toTensor(hr)

        return lr_sample["img"], lr, hr

    def __len__(self):
        return len(self.lr)

    def _init_val(self):
        inputs = self.args.val_file_dir

        self.lr = [{"img": x} for x in sorted(glob.glob(inputs + "/noise/*.png"))]
        self.hr = [{"img": x} for x in sorted(glob.glob(inputs + "/GT/*.png"))]

        print("Total val images : {}".format(len(self.lr)))


class My_AIOTrainDataset(Dataset):
    """
    Dataset class for training on degraded images.
    """

    def __init__(self, args):
        super(My_AIOTrainDataset, self).__init__()
        self.args = args
        self.de_temp = 0
        self.de_type = self.args.de_type
        self.D = Degradation(args)
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        self.de_dict_reverse = {idx: dataset for idx, dataset in enumerate(self.de_type)}

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])
        self.toTensor = ToTensor()

        self._init_lr()
        self._merge_tasks()

    def __getitem__(self, idx):
        lr_sample = self.lr[idx]
        de_id = lr_sample["de_type"]
        deg_type = self.de_dict_reverse[de_id]

        if deg_type == "denoise_15" or deg_type == "denoise_25" or deg_type == "denoise_50":

            hr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
            hr = self.crop_transform(hr)
            hr = np.array(hr)

            hr = random_augmentation(hr)[0]
            lr = self.D.single_degrade(hr, de_id)
        else:
            if deg_type == "dehaze":
                lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
                clean_name = self._get_nonhazy_name(lr_sample["img"])
                hr = crop_img(np.array(Image.open(clean_name).convert('RGB')), base=16)

            else:
                hr_sample = self.hr[idx]
                lr = crop_img(np.array(Image.open(lr_sample["img"]).convert('RGB')), base=16)
                hr = crop_img(np.array(Image.open(hr_sample["img"]).convert('RGB')), base=16)

            lr, hr = random_augmentation(*self._crop_patch(lr, hr))

            # import matplotlib.pyplot as plt
            # import os
            # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            # plt.imshow(lr)
            # plt.show()
            # plt.imshow(hr)
            # plt.show()

        lr = self.toTensor(lr)
        hr = self.toTensor(hr)

        return [lr_sample["img"], de_id], lr, hr

    def __len__(self):
        return len(self.lr)

    def _init_lr(self):
        # synthetic datasets
        if 'lowlight' in self.de_type:
            self._init_lowlight(id=self.de_dict['lowlight'])
        if 'deblur' in self.de_type:
            self._init_deblur(id=self.de_dict['deblur'])
        if 'sidd' in self.de_type:
            self._init_SIDD(id=self.de_dict['sidd'])
        if 'nind' in self.de_type:
            self._init_NIND(id=self.de_dict['nind'])
        if 'derain' in self.de_type:
            self._init_derain(id=self.de_dict['derain'])
        if 'dehaze' in self.de_type:
            self._init_dehaze(id=self.de_dict['dehaze'])
        if 'denoise_15' in self.de_type:
            self._init_clean(id=0)
        if 'denoise_25' in self.de_type:
            self._init_clean(id=0)
        if 'denoise_50' in self.de_type:
            self._init_clean(id=0)

    def _merge_tasks(self):
        self.lr = []
        self.hr = []
        # synthetic datasets
        if "lowlight" in self.de_type:
            self.lr += self.synllie_lr
            self.hr += self.synllie_hr
        if "deblur" in self.de_type:
            self.lr += self.deblur_lr
            self.hr += self.deblur_hr
        if "sidd" in self.de_type:
            self.lr += self.sidd_lr
            self.hr += self.sidd_hr
        if "nind" in self.de_type:
            self.lr += self.nind_lr
            self.hr += self.nind_hr
        if "denoise_15" in self.de_type:
            self.lr += self.s15_ids
            self.hr += self.s15_ids
        if "denoise_25" in self.de_type:
            self.lr += self.s25_ids
            self.hr += self.s25_ids
        if "denoise_50" in self.de_type:
            self.lr += self.s50_ids
            self.hr += self.s50_ids
        if "derain" in self.de_type:
            self.lr += self.derain_lr
            self.hr += self.derain_hr
        if "dehaze" in self.de_type:
            self.lr += self.dehaze_lr
            self.hr += self.dehaze_hr

        print(len(self.lr))

    def _init_lowlight(self, id):
        datapath = self.args.data_file_dir + "/low-light/lol_dataset"
        csv_path = os.path.join(datapath, "train_pairs.csv")
        # 读取 CSV 文件
        df = pd.read_csv(csv_path)

        # 获取 GT 和 LOL 列的数据
        gt_paths = df['GT'].tolist()
        lol_paths = df['LOL'].tolist()
        self.synllie_lr = [{"img": os.path.join(datapath, x), "de_type": id} for x in lol_paths]
        self.synllie_hr = [{"img": os.path.join(datapath, x), "de_type": id} for x in gt_paths]

        print("Total SynLLIE training pairs : {}".format(len(self.synllie_lr)))
        self.synllie_lr = self.synllie_lr * 10
        self.synllie_hr = self.synllie_hr * 10
        print("Repeated LOL Dataset length : {}".format(len(self.synllie_hr)))

    def _init_deblur(self, id):
        """ Initialize the GoPro training dataset """
        datapath = self.args.data_file_dir + "/deblur/GOPRO_Large"
        csv_path = os.path.join(datapath, "train_pairs.csv")
        # 读取 CSV 文件
        df = pd.read_csv(csv_path)

        gt_paths = df['GT'].tolist()
        blur_paths = df['BLUR'].tolist()

        self.deblur_lr = [{"img": os.path.join(datapath, x), "de_type": id} for x in blur_paths]
        self.deblur_hr = [{"img": os.path.join(datapath, x), "de_type": id} for x in gt_paths]

        print("Total Deblur training pairs : {}".format(len(self.deblur_hr)))
        self.deblur_lr = self.deblur_lr * 3
        self.deblur_hr = self.deblur_hr * 3
        print("Repeated Dataset length : {}".format(len(self.deblur_hr)))

    def _init_SIDD(self, id):
        datapath = self.args.data_file_dir + "/denoising/SIDD"
        csv_path = os.path.join(datapath, "SIDD_train_crops.csv")
        # 读取 CSV 文件
        df = pd.read_csv(csv_path)

        gt_paths = df['Clean Image:FILE'].tolist()
        noise_paths = df['Noisy Image:FILE'].tolist()

        self.sidd_lr = [{"img": os.path.join(datapath, x), "de_type": id} for x in noise_paths]
        self.sidd_hr = [{"img": os.path.join(datapath, x), "de_type": id} for x in gt_paths]

        print("Total SIDD training pairs : {}".format(len(self.sidd_lr)))

    def _init_NIND(self, id):
        datapath = self.args.data_file_dir + "/denoising/NIND"
        csv_path = os.path.join(datapath, "pairs.csv")
        # 读取 CSV 文件
        df = pd.read_csv(csv_path)

        gt_paths = df['GT'].tolist()
        noise_paths = df['NOISE'].tolist()

        self.nind_lr = [{"img": os.path.join(datapath, x), "de_type": id} for x in noise_paths]
        self.nind_hr = [{"img": os.path.join(datapath, x), "de_type": id} for x in gt_paths]

        print("Total NIND training pairs : {}".format(len(self.nind_lr)))

        self.nind_lr = self.nind_lr * 10
        self.nind_hr = self.nind_hr * 10
        print("Repeated NIND Dataset length : {}".format(len(self.nind_lr)))

    def _init_derain(self, id):
        inputs = self.args.data_file_dir + "/deraining/RainTrainL/rainy"
        targets = self.args.data_file_dir + "/deraining/RainTrainL/gt"

        self.derain_lr = [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + "/*.png"))]
        self.derain_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.png"))]

        self.derain_counter = 0
        print("Total Derain training pairs : {}".format(len(self.derain_lr)))
        self.derain_lr = self.derain_lr * 120
        self.derain_hr = self.derain_hr * 120
        print("Repeated Dataset length : {}".format(len(self.derain_hr)))

    def _init_dehaze(self, id):
        inputs = self.args.data_file_dir + "/dehazing/RESIDE/"
        targets = self.args.data_file_dir + "/dehazing/RESIDE/clear"

        self.dehaze_lr = []
        for part in ["part1", "part2", "part3", "part4"]:
            self.dehaze_lr += [{"img": x, "de_type": id} for x in sorted(glob.glob(inputs + part + "/*.jpg"))]

        self.dehaze_hr = [{"img": x, "de_type": id} for x in sorted(glob.glob(targets + "/*.jpg"))]

        self.dehaze_counter = 0
        print("Total Dehaze training pairs : {}".format(len(self.dehaze_lr)))
        self.dehaze_lr = self.dehaze_lr
        self.dehaze_hr = self.dehaze_hr
        print("Repeated Dataset length : {}".format(len(self.dehaze_lr)))

    def _init_clean(self, id):
        inputs = self.args.data_file_dir + "/denoising"

        clean = []
        for dataset in ["WaterlooED", "BSD400"]:
            if dataset == "WaterlooED":
                ext = "bmp"
            else:
                ext = "jpg"
            clean += [x for x in sorted(glob.glob(inputs + f"/{dataset}/*.{ext}"))]

        if 'denoise_15' in self.de_type:
            self.s15_ids = [{"img": x, "de_type": self.de_dict['denoise_15']} for x in clean]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if 'denoise_25' in self.de_type:
            self.s25_ids = [{"img": x, "de_type": self.de_dict['denoise_25']} for x in clean]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if 'denoise_50' in self.de_type:
            self.s50_ids = [{"img": x, "de_type": self.de_dict['denoise_50']} for x in clean]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _get_nonhazy_name(self, hazy_name):
        dir_name = os.path.dirname(os.path.dirname(hazy_name)) + "/clear"
        name = hazy_name.split('/')[-1].split('_')[0]
        suffix = os.path.splitext(hazy_name)[1]
        nonhazy_name = dir_name + "/" + name + suffix
        return nonhazy_name


class H5_AIOTrainDataset_2(Dataset):
    """
    Optimized Dataset class for training on degraded images with HDF5 storage.
    Stores only original data without duplication, including shared REDS GT.
    """

    def __init__(self, args):
        super(H5_AIOTrainDataset_2, self).__init__()
        seed_everything(42)
        self.args = args
        self.de_temp = 0
        self.de_type = self.args.de_type
        self.D = Degradation(args)
        self.de_dict = {dataset: idx for idx, dataset in enumerate(self.de_type)}
        self.de_dict_reverse = {idx: dataset for idx, dataset in enumerate(self.de_type)}

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(args.patch_size),
        ])
        self.toTensor = ToTensor()

        # Initialize HDF5 file
        self.hdf5_path = args.hdf5_path
        # self.hdf5_path = os.path.join('H:/datasets/IR/all_data_optimized.h5')
        self._init_or_update_hdf5_file()

        # Open HDF5 file in read mode
        self.hdf5_file = h5pickle.File(self.hdf5_path, 'r')

        # Initialize sample indices
        self._init_sample_indices()

    def __getitem__(self, idx):
        sample_info = self.sample_indices[idx]
        de_id = sample_info["de_type"]
        deg_type = self.de_dict_reverse[de_id]
        data_idx = sample_info["data_idx"]
        is_denoise = deg_type in ["denoise_15", "denoise_25", "denoise_50", "poisson_1"]

        # Get data from HDF5
        if is_denoise:
            group = self.hdf5_file[f'clean/{data_idx}']
            hr = np.array(group['hr'])
            hr = self.crop_transform(hr)
            hr = np.array(hr)
            hr = random_augmentation(hr)[0]
            lr = self.D.single_degrade(hr, deg_type)
        else:
            # For REDS datasets, get HR from shared group
            if deg_type in ["redblur", "redblurjpeg", "redblurcomp"]:
                lr_group = self.hdf5_file[f'{deg_type}/{data_idx}']
                hr_group = self.hdf5_file[f'reds_gt/{sample_info["gt_idx"]}']
                lr = np.array(lr_group['lr'])
                hr = np.array(hr_group['hr'])
            else:
                group = self.hdf5_file[f'{deg_type}/{data_idx}']
                lr = np.array(group['lr'])
                hr = np.array(group['hr'])
            lr, hr = random_augmentation(*self._crop_patch(lr, hr))

        lr = self.toTensor(lr)
        hr = self.toTensor(hr)

        return [sample_info["img_path"], de_id], lr, hr

    def __len__(self):
        return len(self.sample_indices)

    def _init_sample_indices(self):
        """Initialize sample indices without duplicating data"""
        self.sample_indices = []

        # For each degradation type, create references to original data
        for deg_type in self.de_type:
            de_id = self.de_dict[deg_type]

            if deg_type in ["denoise_15", "denoise_25", "denoise_50", "poisson_1"]:
                # For denoise tasks, we'll use the same clean images multiple times
                num_clean = len(self.hdf5_file['clean'])
                repeat_factor = 1
                for _ in range(repeat_factor):
                    for i in range(num_clean):
                        group = self.hdf5_file[f'clean/{i}']
                        self.sample_indices.append({
                            "de_type": de_id,
                            "data_idx": i,
                            "img_path": group.attrs['img_path']
                        })
                print(f"Number of {deg_type} samples: {num_clean} (Repeats: {repeat_factor})")
            elif deg_type in ["redblur", "redblurjpeg", "redblurcomp"]:
                # For REDS datasets, we'll reference the shared GT
                num_samples = len(self.hdf5_file[deg_type])
                for i in range(num_samples):
                    group = self.hdf5_file[f'{deg_type}/{i}']
                    gt_idx = group.attrs['gt_idx']  # Get the shared GT index
                    self.sample_indices.append({
                        "de_type": de_id,
                        "data_idx": i,
                        "gt_idx": gt_idx,
                        "img_path": group.attrs['img_path']
                    })
                print(f"Number of {deg_type} samples: {num_samples}")
            else:
                # For other tasks, use each sample once
                num_samples = len(self.hdf5_file[deg_type])
                repeat_factor = 8 if deg_type in ["lowlight", "nind"] else 3 if deg_type == "deblur" else 1
                if deg_type == "zte":
                    repeat_factor = 200
                if deg_type == "polyu":
                    repeat_factor = 40
                print(f"Number of {deg_type} samples: {num_samples} (Repeats: {repeat_factor})")
                for _ in range(repeat_factor):
                    if deg_type == "sidd":
                        # TODO
                        # Randomly select 60% of the samples
                        all_indices = list(range(num_samples))
                        random.shuffle(all_indices)
                        selected_indices = all_indices[:int(num_samples * 1.0)]
                        for i in selected_indices:
                            group = self.hdf5_file[f'{deg_type}/{i}']
                            self.sample_indices.append({
                                "de_type": de_id,
                                "data_idx": i,
                                "img_path": group.attrs['img_path']
                            })
                    else:
                        for i in range(num_samples):
                            group = self.hdf5_file[f'{deg_type}/{i}']
                            self.sample_indices.append({
                                "de_type": de_id,
                                "data_idx": i,
                                "img_path": group.attrs['img_path']
                            })

        # Shuffle the indices
        random.shuffle(self.sample_indices)
        print(f"Total training samples: {len(self.sample_indices)}")

    def _init_or_update_hdf5_file(self, override=False):
        """Initialize or update HDF5 file, only adding missing datasets"""
        if not os.path.exists(self.hdf5_path):
            print("Creating new HDF5 file...")
            self._create_hdf5_file()
            return

        print("HDF5 file exists, checking for missing datasets...")
        with h5pickle.File(self.hdf5_path, 'a') as hf:
            needs_update = False

            # Check if clean group is needed and exists
            need_clean = any(t in self.de_type for t in ["denoise_15", "denoise_25", "denoise_50","poisson_1"])
            if need_clean and 'clean' not in hf:
                print("Adding missing clean dataset...")
                self._add_clean_dataset(hf)
                needs_update = True

            # Check if REDS GT group is needed
            reds_types = ["redblur", "redblurjpeg", "redblurcomp"]
            need_reds_gt = any(t in self.de_type for t in reds_types)
            if need_reds_gt and 'reds_gt' not in hf:
                print("Adding missing REDS GT dataset...")
                self._add_reds_gt_dataset(hf)
                needs_update = True

            # Check for missing degradation types
            for deg_type in self.de_type:
                if deg_type in ["denoise_15", "denoise_25", "denoise_50","poisson_1"] + reds_types:
                    continue

                if deg_type not in hf:
                    print(f"Adding missing {deg_type} dataset...")
                    self._add_degradation_dataset(hf, deg_type)
                    needs_update = True

            # Check for missing REDS degradation types
            for deg_type in reds_types:
                if deg_type in self.de_type and deg_type not in hf:
                    print(f"Adding missing {deg_type} dataset...")
                    self._add_reds_degradation_dataset(hf, deg_type)
                    needs_update = True

        if not needs_update:
            print("All required datasets already exist in HDF5 file.")

    def _create_hdf5_file(self):
        """Create new HDF5 file with all datasets"""
        print("Creating optimized HDF5 file for faster data loading...")
        self._init_dataset_paths()

        with h5pickle.File(self.hdf5_path, 'w') as hf:
            # Store clean images (for denoising tasks)
            if any(t in self.de_type for t in ["denoise_15", "denoise_25", "denoise_50","poisson_1"]):
                self._add_clean_dataset(hf)

            # Store REDS GT if needed
            reds_types = ["redblur", "redblurjpeg", "redblurcomp"]
            if any(t in self.de_type for t in reds_types):
                self._add_reds_gt_dataset(hf)

            # Store other datasets
            for deg_type in self.de_type:
                if deg_type in ["denoise_15", "denoise_25", "denoise_50", "poisson_1"]:
                    continue
                if deg_type in reds_types:
                    self._add_reds_degradation_dataset(hf, deg_type)
                else:
                    self._add_degradation_dataset(hf, deg_type)

        print("Optimized HDF5 file created successfully!")

    def _add_clean_dataset(self, hf):
        """Add clean dataset to HDF5 file"""
        clean_group = hf.create_group('clean')
        inputs = self.args.data_file_dir + "/denoising"

        clean_paths = []
        for dataset in ["WaterlooED", "BSD400"]:
            ext = "bmp" if dataset == "WaterlooED" else "jpg"
            clean_paths += sorted(glob.glob(inputs + f"/{dataset}/*.{ext}"))

        for i, img_path in enumerate(tqdm.tqdm(clean_paths, desc="Storing clean images")):
            group = clean_group.create_group(str(i))
            hr = crop_img(np.array(Image.open(img_path).convert('RGB')), base=16)
            group.create_dataset('hr', data=hr)
            group.attrs['img_path'] = img_path

    def _add_reds_gt_dataset(self, hf):
        """Add shared REDS GT dataset to HDF5 file"""
        gt_group = hf.create_group('reds_gt')
        datapath = self.args.data_file_dir + "/REDS"

        # Get all unique GT paths from REDS datasets
        gt_paths = set()
        for csv_file in ['RED_Blur.csv', 'RED_BlurJpeg.csv', 'RED_BlurComp.csv']:
            csv_path = os.path.join(datapath, csv_file)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                gt_paths.update(df['GT'].tolist())

        # Store each unique GT image
        for i, gt_path in enumerate(tqdm.tqdm(sorted(gt_paths), desc="Storing REDS GT images")):
            group = gt_group.create_group(str(i))
            hr = crop_img(np.array(Image.open(os.path.join(datapath, gt_path)).convert('RGB')), base=16)
            group.create_dataset('hr', data=hr)
            group.attrs['img_path'] = gt_path

    # 定义一个函数来提取路径的尾部两级路径
    def get_last_two_levels_without_ext(self,path):
        # 先去掉文件后缀
        path_without_ext = os.path.splitext(path)[0]
        parts = []
        # 循环两次提取尾部两级路径
        for _ in range(2):
            path_without_ext, tail = os.path.split(path_without_ext)
            if tail:
                parts.insert(0, tail)
        return os.path.join(*parts)

    def _add_reds_degradation_dataset(self, hf, deg_type):
        """Add REDS degradation dataset with references to shared GT"""
        try:
            dataset_group = hf.create_group(deg_type)
            csv_file = f'RED_{deg_type[3:].capitalize()}.csv'  # redblur -> RED_Blur.csv
            paths = self._get_REDBlur_paths(csv_file)

            # First build a mapping from GT path to its index in reds_gt group
            # gt_path_to_idx = {gt.attrs['img_path']: idx for idx, gt in enumerate(hf['reds_gt'].values())}

            # 对 hf['reds_gt'].values() 按 img_path 排序
            sorted_gt_values = sorted(hf['reds_gt'].values(), key=lambda gt: gt.attrs['img_path'])
            gt_path_to_idx = {gt.attrs['img_path']: idx for idx, gt in enumerate(sorted_gt_values)}

            for i, (lr_path, hr_path) in enumerate(tqdm.tqdm(paths, desc=f"Storing {deg_type} pairs")):
                group = dataset_group.create_group(str(i))
                lr_img = crop_img(np.array(Image.open(lr_path).convert('RGB')), base=16)
                group.create_dataset('lr', data=lr_img)
                group.attrs['img_path'] = lr_path
                # 截取REDS后面的路径部分
                hr_path = hr_path.split('REDS')[1].lstrip(os.sep)
                # print(hf['reds_gt'][str(gt_path_to_idx[hr_path])].attrs['img_path'],lr_path)
                # 确保尾部两级路径一致
                last_two_levels_1 = self.get_last_two_levels_without_ext(hf['reds_gt'][str(gt_path_to_idx[hr_path])].attrs['img_path'])
                last_two_levels_2 = self.get_last_two_levels_without_ext(lr_path)
                assert last_two_levels_1 == last_two_levels_2, f"去掉后缀后的尾部两级路径不一致，路径1: {last_two_levels_1}，路径2: {last_two_levels_2}"
                group.attrs['gt_idx'] = gt_path_to_idx[hr_path]  # Store reference to shared GT

        except Exception as e:
            print(f"Error adding dataset: {deg_type}")
            print(e)
            if deg_type in hf:
                del hf[deg_type]

    def _add_degradation_dataset(self, hf, deg_type):
        """Add specific degradation dataset to HDF5 file"""
        try:
            dataset_group = hf.create_group(deg_type)

            if deg_type == "lowlight":
                paths = self._get_lowlight_paths()
            elif deg_type == "deblur":
                paths = self._get_deblur_paths()
            elif deg_type == "sidd":
                paths = self._get_sidd_paths()
            elif deg_type == "nind":
                paths = self._get_nind_paths()
            elif deg_type == "zte":
                paths = self._get_ZTE_Example_paths()
            elif deg_type == "polyu":
                paths = self._get_POLYU_paths()

            for i, (lr_path, hr_path) in enumerate(tqdm.tqdm(paths, desc=f"Storing {deg_type} pairs")):
                group = dataset_group.create_group(str(i))
                lr_img = crop_img(np.array(Image.open(lr_path).convert('RGB')), base=16)
                hr_img = crop_img(np.array(Image.open(hr_path).convert('RGB')), base=16)
                group.create_dataset('lr', data=lr_img)
                group.create_dataset('hr', data=hr_img)
                group.attrs['img_path'] = lr_path
        except Exception as e:
            print(f"Error adding dataset: {deg_type}")
            print(e)
            if deg_type in hf:
                del hf[deg_type]

    def _init_dataset_paths(self):
        """Initialize dataset paths without creating duplicate lists"""
        pass

    def _get_lowlight_paths(self):
        datapath = self.args.data_file_dir + "/low-light/lol_dataset"
        csv_path = os.path.join(datapath, "train_pairs.csv")
        df = pd.read_csv(csv_path)
        return [(os.path.join(datapath, x), os.path.join(datapath, y))
                for x, y in zip(df['LOL'].tolist(), df['GT'].tolist())]

    def _get_deblur_paths(self):
        datapath = self.args.data_file_dir + "/deblur/GOPRO_Large"
        csv_path = os.path.join(datapath, "train_pairs.csv")
        df = pd.read_csv(csv_path)
        return [(os.path.join(datapath, x), os.path.join(datapath, y))
                for x, y in zip(df['BLUR'].tolist(), df['GT'].tolist())]

    def _get_sidd_paths(self):
        datapath = self.args.data_file_dir + "/denoising/SIDD"
        csv_path = os.path.join(datapath, "SIDD_train_crops.csv")
        df = pd.read_csv(csv_path)
        return [(os.path.join(datapath, x), os.path.join(datapath, y))
                for x, y in zip(df['Noisy Image:FILE'].tolist(), df['Clean Image:FILE'].tolist())]

    def _get_nind_paths(self):
        datapath = self.args.data_file_dir + "/denoising/NIND"
        csv_path = os.path.join(datapath, "pairs.csv")
        df = pd.read_csv(csv_path)
        return [(os.path.join(datapath, x), os.path.join(datapath, y))
                for x, y in zip(df['NOISE'].tolist(), df['GT'].tolist())]

    def _get_ZTE_Example_paths(self):
        datapath = 'D:/1_Image_Denoising/ZTE_Example'
        lr_files = sorted(glob.glob(os.path.join(datapath, "noise/*.png")))
        hr_files = sorted(glob.glob(os.path.join(datapath, "GT/*.png")))
        return [(x, y) for x, y in zip(lr_files, hr_files)]

    def _get_REDBlur_paths(self, csv_file):
        datapath = self.args.data_file_dir + "/REDS"
        csv_path = os.path.join(datapath, csv_file)
        df = pd.read_csv(csv_path)
        return [(os.path.join(datapath, x), os.path.join(datapath, y))
                for x, y in zip(df['BLUR'].tolist(), df['GT'].tolist())]

    def _get_POLYU_paths(self):
        datapath = self.args.data_file_dir + "/denoising/PolyU"
        csv_path = os.path.join(datapath, "pairs.csv")
        df = pd.read_csv(csv_path)
        return [(os.path.join(datapath, x), os.path.join(datapath, y))
                for x, y in zip(df['NOISE'].tolist(), df['GT'].tolist())]

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

if __name__ == "__main__":
    from src.options import train_options
    opt = train_options()
    trainset = H5_AIOTrainDataset_2(opt)
    print(trainset)

    # 获取一个元素
    index = 1000
    [lr_sample, de_id], lr, hr = trainset[index]

    print(lr_sample)
    print(de_id)
    print(len(lr))
    print(len(hr))
    print(lr.shape)
    print(hr.shape)

    from skimage.util import img_as_ubyte
    cv2.imwrite("lr.png", cv2.cvtColor(img_as_ubyte(lr.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))
    cv2.imwrite("hr.png", cv2.cvtColor(img_as_ubyte(hr.permute(1, 2, 0).numpy()), cv2.COLOR_RGB2BGR))