import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import cv2

def display_comparison(image_names, noisy_dir, denoised_dir, n_rows=1, patch_coords_list=None, patch_size=(256, 256)):
    """
    展示NOISY和Denoised图片对比

    参数:
        image_names: 图片文件名列表(不带路径)
        noisy_dir: NOISY图片目录
        denoised_dir: Denoised图片目录
        n_rows: 显示行数
        patch_coords_list: [(x1,y1), (x2,y2)...] 每张图的截取区域左上角坐标列表
        patch_size: (width,height) 统一的截取块大小
    """
    n_images = len(image_names)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row in range(n_rows):
        for col in range(4):
            idx = row * 2 + col // 2  # 每行显示2组对比(4列)
            if idx >= n_images:
                break

            img_name = image_names[idx]
            # 奇数列显示NOISY，偶数列显示Denoised
            img_path = os.path.join(noisy_dir if col % 2 == 0 else denoised_dir, img_name)

            if not os.path.exists(img_path):
                print(f"图片不存在: {img_path}")
                continue

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 如果指定了截取区域
            if patch_coords_list and idx < len(patch_coords_list) and patch_size:
                x, y = patch_coords_list[idx]
                w, h = patch_size
                img = img[y:y+h, x:x+w]

            ax = axes[row, col]
            ax.imshow(img)
            ax.set_title(f"{'NOISY' if col % 2 == 0 else 'Denoised'}: {img_name}")
            ax.axis('off')

    plt.tight_layout(h_pad=3.0, w_pad=0.5)  # 增加行间距(h_pad)，减小列间距(w_pad)
    plt.show()

if __name__ == "__main__":
    # 设置目录路径
    noisy_dir = r"D:\1_Image_Denoising\ZTE_Data"
    denoised_dir = r"D:\1_Image_Denoising\MOE-IR-src\results\onnx"

    # 手动设置要显示的图片文件名
    image_names = ["00004.png", "00020.png", "00029.png", "00085.png", "00111.png", "00105.png", "00055.png", "00106.png"]

    # 为每张图设置不同的起始坐标，但使用统一的截取大小
    patch_coords_list = [(100, 100), (300, 400), (700, 1050), (100, 100), (200, 400), (500, 400), (500, 800), (400, 800)]
    patch_size = (256, 256)  # 所有截取块统一大小

    # 显示图片对比
    display_comparison(image_names, noisy_dir, denoised_dir,
                      n_rows=4,
                      patch_coords_list=patch_coords_list,
                      patch_size=patch_size)