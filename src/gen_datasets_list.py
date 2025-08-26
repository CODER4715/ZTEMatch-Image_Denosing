
import os
import pandas as pd

def generate_GoPro_csv(dataset_root, output_csv, dataset_type):
    data = []
    # 遍历二级目录
    for sub_dir in os.listdir(dataset_root):
        sub_dir_path = os.path.join(dataset_root, sub_dir)
        if os.path.isdir(sub_dir_path):
            gt_dir = os.path.join(sub_dir_path, 'sharp')
            noise_dir = os.path.join(sub_dir_path, 'blur_gamma')
            # 检查GT和NOISE目录是否存在
            if os.path.exists(gt_dir) and os.path.exists(noise_dir):
                gt_files = sorted(os.listdir(gt_dir))
                noise_files = sorted(os.listdir(noise_dir))
                # 确保GT和NOISE图片数量一致
                if len(gt_files) == len(noise_files):
                    for gt_file, noise_file in zip(gt_files, noise_files):
                        gt_path = os.path.join(gt_dir, gt_file)
                        noise_path = os.path.join(noise_dir, noise_file)
                        # 获取从 dataset_root 开始的相对路径
                        relative_gt_path = os.path.relpath(gt_path, dataset_root)
                        relative_noise_path = os.path.relpath(noise_path, dataset_root)
                        # 在相对路径前加上数据集类型（train 或 test）
                        relative_gt_path = os.path.join(dataset_type, relative_gt_path)
                        relative_noise_path = os.path.join(dataset_type, relative_noise_path)
                        data.append([relative_gt_path, relative_noise_path])

    df = pd.DataFrame(data, columns=['GT', 'BLUR'])
    df.to_csv(output_csv, index=False)

def generate_haze_csv(dataset_root, output_csv):
    data = []
    # 遍历二级目录
    for sub_dir in os.listdir(dataset_root):
        sub_dir_path = os.path.join(dataset_root, sub_dir)
        if os.path.isdir(sub_dir_path):
            gt_dir = os.path.join(sub_dir_path, 'gt')
            haze_dir = os.path.join(sub_dir_path, 'hazy')
            # 检查 GT 和 HAZE 目录是否存在
            if os.path.exists(gt_dir) and os.path.exists(haze_dir):
                gt_files = os.listdir(gt_dir)
                haze_files = os.listdir(haze_dir)
                for gt_file in gt_files:
                    gt_prefix = os.path.splitext(gt_file)[0]
                    for haze_file in haze_files:
                        haze_prefix = haze_file.split('_')[0]
                        if gt_prefix == haze_prefix:
                            gt_path = os.path.join(gt_dir, gt_file)
                            haze_path = os.path.join(haze_dir, haze_file)
                            relative_gt_path = os.path.relpath(gt_path, dataset_root)
                            relative_haze_path = os.path.relpath(haze_path, dataset_root)
                            data.append([relative_gt_path, relative_haze_path])
    # 创建 DataFrame 并保存为 CSV 文件
    df = pd.DataFrame(data, columns=['GT', 'HAZE'])
    df.to_csv(output_csv, index=False)

def generate_lol_csv(dataset_root, output_csv, dataset_type):
    data = []

    gt_dir = os.path.join(dataset_root, 'high')
    noise_dir = os.path.join(dataset_root, 'low')
    # 检查GT和NOISE目录是否存在
    if os.path.exists(gt_dir) and os.path.exists(noise_dir):
        gt_files = sorted(os.listdir(gt_dir))
        noise_files = sorted(os.listdir(noise_dir))
        # 确保GT和NOISE图片数量一致
        if len(gt_files) == len(noise_files):
            for gt_file, noise_file in zip(gt_files, noise_files):
                gt_path = os.path.join(gt_dir, gt_file)
                noise_path = os.path.join(noise_dir, noise_file)
                # 获取从 dataset_root 开始的相对路径
                relative_gt_path = os.path.relpath(gt_path, dataset_root)
                relative_noise_path = os.path.relpath(noise_path, dataset_root)
                # 在相对路径前加上数据集类型（train 或 test）
                relative_gt_path = os.path.join(dataset_type, relative_gt_path)
                relative_noise_path = os.path.join(dataset_type, relative_noise_path)
                data.append([relative_gt_path, relative_noise_path])

    df = pd.DataFrame(data, columns=['GT', 'LOL'])
    df.to_csv(output_csv, index=False)

def generate_NIND_csv(dataset_root, output_csv):
    data = []
    # 遍历所有文件夹
    for folder in os.listdir(dataset_root):
        folder_path = os.path.join(dataset_root, folder)
        if os.path.isdir(folder_path):
            # 获取当前文件夹下的所有图片文件
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            # 分别存储数值和非数值的 ISO 标识及其对应的文件名
            num_iso_values = {}
            non_num_iso_values = {}
            for image_file in image_files:
                iso_start = image_file.rfind('ISO')
                if iso_start != -1:
                    iso_str = image_file[iso_start + 3:image_file.rfind('.')]
                    try:
                        iso_value = int(iso_str)
                        num_iso_values[iso_value] = image_file
                    except ValueError:
                        non_num_iso_values[iso_str] = image_file
            # 找到最小的数值 ISO 值对应的图片作为 GT
            if num_iso_values:
                min_iso = min(num_iso_values.keys())
                gt_file = num_iso_values[min_iso]
                gt_path = os.path.join(folder_path, gt_file)
                # 遍历其他数值 ISO 值和非数值 ISO 标识的图片作为 NOISE
                all_noise_files = {**{k: v for k, v in num_iso_values.items() if k != min_iso}, **non_num_iso_values}
                for noise_file in all_noise_files.values():
                    noise_path = os.path.join(folder_path, noise_file)
                    relative_gt_path = os.path.relpath(gt_path, dataset_root)
                    relative_noise_path = os.path.relpath(noise_path, dataset_root)
                    data.append([relative_gt_path, relative_noise_path])
    # 创建 DataFrame 并保存为 CSV 文件
    df = pd.DataFrame(data, columns=['GT', 'NOISE'])
    df.to_csv(output_csv, index=False)

def generate_REDBlur_csv(dataset_root_blur, dataset_root_sharp, output_csv):
    data = []
    base_dir = os.path.dirname(os.path.dirname(dataset_root_blur))  # 获取 G:\datasets\IR\REDS
    # 遍历 train_blur 的二级目录
    for sub_dir in os.listdir(dataset_root_blur):
        sub_dir_path = os.path.join(dataset_root_blur, sub_dir)
        if os.path.isdir(sub_dir_path):
            # 获取当前二级目录下的所有模糊图片文件
            blur_files = sorted([f for f in os.listdir(sub_dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            # 对应的 sharp 目录
            sharp_sub_dir_path = os.path.join(dataset_root_sharp, sub_dir)
            if os.path.exists(sharp_sub_dir_path):
                # 获取当前二级目录下的所有清晰图片文件
                sharp_files = sorted([f for f in os.listdir(sharp_sub_dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
                # 确保模糊和清晰图片数量一致
                if len(blur_files) == len(sharp_files):
                    for blur_file, sharp_file in zip(blur_files, sharp_files):
                        blur_path = os.path.join(sub_dir_path, blur_file)
                        sharp_path = os.path.join(sharp_sub_dir_path, sharp_file)
                        # 获取从 G:\datasets\IR\REDS 开始的相对路径
                        relative_blur_path = os.path.relpath(blur_path, base_dir)
                        relative_sharp_path = os.path.relpath(sharp_path, base_dir)
                        data.append([relative_sharp_path, relative_blur_path])

    df = pd.DataFrame(data, columns=['GT', 'BLUR'])
    df.to_csv(output_csv, index=False)


def generate_polyu_csv(dataset_root, output_csv):
    data = []
    # 获取CroppedImages的父目录作为基准路径
    base_dir = os.path.dirname(dataset_root)
    # 遍历数据集目录
    for filename in os.listdir(dataset_root):
        if filename.endswith('_real.JPG'):
            # 获取对应的 GT 文件名（将 real 替换为 mean）
            gt_filename = filename.replace('_real.JPG', '_mean.JPG')
            gt_path = os.path.join(dataset_root, gt_filename)
            noise_path = os.path.join(dataset_root, filename)

            # 检查 GT 文件是否存在
            if os.path.exists(gt_path):
                # 获取从base_dir开始的相对路径（包含CroppedImages）
                relative_gt_path = os.path.relpath(gt_path, base_dir)
                relative_noise_path = os.path.relpath(noise_path, base_dir)
                data.append([relative_gt_path, relative_noise_path])

    # 创建 DataFrame 并保存为 CSV 文件
    df = pd.DataFrame(data, columns=['GT', 'NOISE'])
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    dataset_root_prefix = r'G:\datasets\IR'

    # Gopro
    dataset_root = os.path.join(dataset_root_prefix, 'deblur', 'GOPRO_Large', 'train')
    output_csv = os.path.join(dataset_root_prefix, 'deblur', 'GOPRO_Large', 'train_pairs.csv')
    generate_GoPro_csv(dataset_root, output_csv, 'train')
    # 测试集根目录
    dataset_root = os.path.join(dataset_root_prefix, 'deblur', 'GOPRO_Large', 'test')
    output_csv = os.path.join(dataset_root_prefix, 'deblur', 'GOPRO_Large', 'test_pairs.csv')
    generate_GoPro_csv(dataset_root, output_csv, 'test')

    # Dehaze/SOTS
    dataset_root = os.path.join(dataset_root_prefix, 'dehaze', 'SOTS')
    output_csv = os.path.join(dataset_root_prefix, 'dehaze', 'SOTS', 'pairs.csv')
    generate_haze_csv(dataset_root, output_csv)

    # LOL
    dataset_root = os.path.join(dataset_root_prefix, 'low-light', 'lol_dataset', 'train')
    output_csv = os.path.join(dataset_root_prefix, 'low-light', 'lol_dataset', 'train_pairs.csv')
    generate_lol_csv(dataset_root, output_csv, 'train')
    # 测试集根目录
    dataset_root = os.path.join(dataset_root_prefix, 'low-light', 'lol_dataset', 'test')
    output_csv = os.path.join(dataset_root_prefix, 'low-light', 'lol_dataset', 'test_pairs.csv')
    generate_lol_csv(dataset_root, output_csv, 'test')

    # NIND
    # 数据集根目录
    dataset_root = os.path.join(dataset_root_prefix, 'denoising', 'NIND')
    # 输出 CSV 文件路径
    output_csv = os.path.join(dataset_root_prefix, 'denoising', 'NIND', 'pairs.csv')
    generate_NIND_csv(dataset_root, output_csv)

    # REDS
    dataset_root_blur = os.path.join(dataset_root_prefix, 'REDS', 'train', 'train_blur')
    dataset_root_sharp = os.path.join(dataset_root_prefix, 'REDS', 'train', 'train_sharp')
    output_csv = os.path.join(dataset_root_prefix, 'REDS', 'RED_Blur.csv')
    generate_REDBlur_csv(dataset_root_blur, dataset_root_sharp, output_csv)

    dataset_root_blur = os.path.join(dataset_root_prefix, 'REDS', 'train', 'train_blur_jpeg')
    dataset_root_sharp = os.path.join(dataset_root_prefix, 'REDS', 'train', 'train_sharp')
    output_csv = os.path.join(dataset_root_prefix, 'REDS', 'RED_BlurJpeg.csv')
    generate_REDBlur_csv(dataset_root_blur, dataset_root_sharp, output_csv)

    dataset_root_blur = os.path.join(dataset_root_prefix, 'REDS', 'train', 'train_blur_comp')
    dataset_root_sharp = os.path.join(dataset_root_prefix, 'REDS', 'train', 'train_sharp')
    output_csv = os.path.join(dataset_root_prefix, 'REDS', 'RED_BlurComp.csv')
    generate_REDBlur_csv(dataset_root_blur, dataset_root_sharp, output_csv)

    # PolyU
    dataset_root = os.path.join(dataset_root_prefix, 'denoising', 'PolyU', 'CroppedImages')
    output_csv = os.path.join(dataset_root_prefix, 'denoising', 'PolyU', 'pairs.csv')
    generate_polyu_csv(dataset_root, output_csv)
