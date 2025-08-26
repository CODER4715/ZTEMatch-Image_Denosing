import onnxruntime as ort
import os
import pathlib
import argparse
import numpy as np
from PIL import Image
import random

from tqdm import tqdm
from skimage.util import img_as_ubyte

import cv2
import glob

# def save_img(filepath, img):
#     cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
def save_img(filepath, img):
    try:
        ext = os.path.splitext(filepath)[1]
        success, buf = cv2.imencode(ext, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if success:
            with open(filepath, 'wb') as f:
                f.write(buf.tobytes())
        else:
            raise Exception("Image encoding failed")
    except Exception as e:
        raise Exception(f"Save image failed: {filepath}, Error: {str(e)}")

def get_inputs(args):
    inputs = args.data_file_dir
    # 支持的图像格式扩展名
    image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif')
    # 使用glob.glob匹配多种格式
    lr = []
    for ext in image_extensions:
        lr.extend(glob.glob(os.path.join(inputs, ext)))
    lr = [{"img": x} for x in sorted(lr)]
    print("Total inference images : {}".format(len(lr)))

    return lr

def read_img(img_path):
    image = np.array(Image.open(img_path).convert('RGB'))
    # image = cv2.GaussianBlur(image, (3, 3), 0)
    # image = cv2.bilateralFilter(image, d=9, sigmaColor=50, sigmaSpace=50)
    image = image.astype(np.float32) / 255
    h = image.shape[0]
    w = image.shape[1]
    clean_name = img_path
    image = np.transpose(image, (2, 0, 1))
    return clean_name, image, h, w


def resize_tensor(tensor, size):
    """
    调整4D张量的后两维尺寸 (H, W)

    参数:
    tensor: 形状为(1, 3, h, w)的张量
    size: 目标尺寸 (h, w)

    返回:
    调整大小后的张量
    """
    print("resize")
    # 转换为通道在后的格式 (1, h, w, 3) 便于cv2处理
    tensor = np.transpose(tensor, (0, 2, 3, 1))

    # 调整大小
    resized = cv2.resize(tensor[0], (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

    # 如果输入是单批次，添加批次维度
    if len(tensor.shape) == 4:
        resized = np.expand_dims(resized, axis=0)

    # 转换回通道在前的格式 (1, 3, h, w)
    resized = np.transpose(resized, (0, 3, 1, 2))

    return resized

def guided_filter_rgb(img, radius=10, eps=0.01, guide_type="gray"):
    """
    对RGB图像进行导向滤波
    :param img: 输入BGR图像
    :param radius: 滤波核半径（类似双边滤波的d）
    :param eps: 正则化参数（越小边缘保持越强）
    :param guide_type: 引导图类型 ("gray"|"self"|"Y")
    """
    # 选择引导图像
    if guide_type == "gray":
        guide = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度引导
    elif guide_type == "Y":
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        guide, _, _ = cv2.split(img_yuv)  # Y通道引导
    else:
        guide = img  # 原图自身引导（BGR三通道独立）

    # 创建滤波器
    gf = cv2.ximgproc.createGuidedFilter(guide, radius, eps)

    # 对BGR三通道分别滤波（避免颜色失真）
    if guide_type == "self":
        return gf.filter(img)  # 直接滤波（可能颜色失真）
    else:
        channels = cv2.split(img)
        filtered = [gf.filter(ch) for ch in channels]
        return cv2.merge(filtered)

def process_and_infer(x, ort_session):
    """
    处理输入图像并使用ONNX Runtime进行推理

    参数:
    x: 输入图像，形状为(1, 3, h, w)的numpy数组
    ort_session: ONNX Runtime会话

    返回:
    result: 推理结果，与输入图像原始尺寸匹配
    """
    _, _, h, w = x.shape
    # 添加输入检查
    assert not np.isnan(x).any(), "输入包含NaN值"
    assert x.min() >= 0 and x.max() <= 1, "输入值超出[0,1]范围"

    # 记录原始尺寸和是否需要转置
    need_transpose = False
    original_size = (h, w)

    # 情况1: 尺寸正好是1080x1920
    if (h, w) == (1080, 1920):
        # 转置为1920x1080
        x = np.transpose(x, (0, 1, 3, 2))
        need_transpose = True
    elif (h, w) == (1920, 1080):
        # 无需处理
        pass
    # 情况2: 尺寸小于1920x1080
    elif h <= 1920 and w <= 1080:
        # 计算需要填充的像素数
        pad_h = 1920 - h
        pad_w = 1080 - w

        # 在高度和宽度上均匀填充0
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # 进行填充 (在H和W维度上填充)
        print("padding")
        x = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                   mode='constant', constant_values=0)

        # 记录填充参数用于后处理
        padding = (pad_top, pad_bottom, pad_left, pad_right)
    # 情况3: 尺寸大于1920x1080
    else:
        # 判断是否需要先转置
        if h > w:
            # 直接resize到1920x1080
            x = resize_tensor(x, (1920, 1080))
        else:
            # 先转置再resize
            x = np.transpose(x, (0, 1, 3, 2))
            x = resize_tensor(x, (1920, 1080))
            need_transpose = True


    # 进行推理
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    result = ort_session.run([output_name], {input_name: x})[0]

    result = np.clip(result, 0, 1)

    # 后处理
    # 情况1: 原始尺寸是1080x1920，需要转置回来
    if need_transpose and original_size == (1080, 1920):
        result = np.transpose(result, (0, 1, 3, 2))
    # 情况2: 之前进行了填充，现在需要裁剪
    elif 'padding' in locals():
        pad_top, pad_bottom, pad_left, pad_right = padding
        # 裁剪掉填充的部分
        result = result[:, :, pad_top:1920 - pad_bottom, pad_left:1080 - pad_right]
    # 情况3: 之前进行了resize，现在需要恢复原始尺寸
    elif original_size not in [(1920, 1080), (1080, 1920)]:
        if need_transpose:
            # 先resize回转置前的尺寸
            result = resize_tensor(result, (original_size[1], original_size[0]))
            # 再转置回来
            result = np.transpose(result, (0, 1, 3, 2))
        else:
            # 直接resize回原始尺寸
            result = resize_tensor(result, original_size)

    result = np.squeeze(np.transpose(result, (0, 2, 3, 1)), axis=0)
    result = np.clip(result, 0, 1)
    result = img_as_ubyte(result)

    return result


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # ONNX Runtime的确定性设置
    ort.set_seed(seed)

def run_main(args):
    opt = args
    # opt = parser.parse_args()
    set_seeds(42)  # 在程序开始时设置所有种子

    print("--------> Inference on", "testset.")
    inputs = get_inputs(opt)
    save_folder = os.path.join(os.getcwd(), f"{opt.output_path}/onnx")
    pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
    print("save to ", save_folder)

    onnx_model_path = "model.onnx"
    # ort_session = ort.InferenceSession(onnx_model_path)
    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # 顺序执行
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(onnx_model_path,
                                       # providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                       sess_options=sess_options)

    for img_path in tqdm(inputs):

        clean_name, degrad_patch, h, w = read_img(img_path['img'])
        degrad_patch = np.expand_dims(degrad_patch, axis=0)

        max_retry = 8
        retry_count = 0
        success = False

        while not success and retry_count < max_retry:
            restored = process_and_infer(degrad_patch, ort_session)
            # TODO 试下不同的参数
            # restored = cv2.GaussianBlur(restored, (3, 3), 0)
            # restored = guided_filter_rgb(restored, radius=10, eps=0.01, guide_type="self")
            restored = cv2.bilateralFilter(restored, d=3, sigmaColor=10, sigmaSpace=10)  # d=3, sigmaColor=10, sigmaSpace=10--22.62

            ext = os.path.splitext(clean_name)[-1]
            save_name = os.path.splitext(os.path.split(clean_name)[-1])[0] + ext
            # save_path = os.path.join(os.getcwd(), f"{opt.output_path}/onnx", save_name)
            save_path = os.path.join(f"{opt.output_path}/onnx", save_name)
            save_img(save_path, restored)

            # 检查文件大小
            if os.path.getsize(save_path) >= 15 * 1024:  # 15KB
                success = True
            else:
                retry_count += 1
                if retry_count < max_retry:
                    print(
                        f"检测到输出异常({os.path.getsize(save_path) / 1024:.1f}KB)，重试推理 ({retry_count}/{max_retry})")
                else:
                    print(f"达到最大重试次数，仍可能存在异常({os.path.getsize(save_path) / 1024:.1f}KB)/{save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file_dir', type=str, default=os.path.join(pathlib.Path.home(), "datasets"),
                        help='Path to datasets.')
    parser.add_argument('--output_path', type=str, default="results/", help='Output save path.')
    args = parser.parse_args()
    run_main(args)
