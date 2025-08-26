import tkinter as tk
from tkinter import filedialog, messagebox
import os, sys
import threading
import onnxruntime as ort
import pathlib
import numpy as np
from PIL import Image
import random

from tqdm import tqdm
from skimage.util import img_as_ubyte

import cv2
import glob


class StdoutRedirector(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def write(self, string):
        try:
            if self.text_widget:  # 检查text_widget是否存在
                self.text_widget.config(state='normal')
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
                self.text_widget.config(state='disabled')
                self.text_widget.update_idletasks()

            # 同时保留原始输出（如果有）
            if self.original_stdout:
                self.original_stdout.write(string)
        except Exception as e:
            if self.original_stdout:
                self.original_stdout.write(f"Redirector error: {str(e)}\n")

    def flush(self):
        if self.original_stdout:
            self.original_stdout.flush()


class TqdmGUI(tqdm):
    def __init__(self, *args, **kwargs):
        # 获取GUI输出文本框引用
        self.gui_text = kwargs.pop('gui_text', None)
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None):
        if self.gui_text:
            self.gui_text.config(state='normal')
            self.gui_text.insert(tk.END, self.format_meter(**self.format_dict) + '\n')
            self.gui_text.see(tk.END)
            self.gui_text.config(state='disabled')
            self.gui_text.update_idletasks()
        else:
            super().display(msg, pos)

class ImageDenoisingGUI:
    def __init__(self, root):
        self.root = root
        self.output_queue = []
        self.is_running = False
        root.title("MOE-IR 图像去噪工具   作者：罗思远@UESTC")

        # 输入路径
        tk.Label(root, text="输入图片路径:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.input_entry = tk.Entry(root, width=50)
        self.input_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(root, text="选择", command=self.select_input_path).grid(row=0, column=2, padx=5, pady=5)
        
        # 输出路径
        tk.Label(root, text="输出结果路径:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.output_entry = tk.Entry(root, width=50)
        self.output_entry.insert(0, os.path.join(os.getcwd(), "results"))
        self.output_entry.grid(row=1, column=1, padx=5, pady=5)
        tk.Button(root, text="选择", command=self.select_output_path).grid(row=1, column=2, padx=5, pady=5)
        
        # 按钮区域
        button_frame = tk.Frame(root)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        self.run_button = tk.Button(button_frame, text="开始推理", command=self.run_inference, width=15)
        self.run_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(button_frame, text="中断", command=self.stop_inference,
                                    width=15, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)

        self.open_button = tk.Button(button_frame, text="打开输出目录", command=self.open_output_dir, width=15)
        self.open_button.pack(side=tk.LEFT, padx=10)

        # 添加输出显示区域
        output_frame = tk.Frame(root)  # 确保这行代码存在
        output_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        self.output_text = tk.Text(output_frame, height=15, state='disabled')
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(output_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.output_text.yview)

        sys.stdout = StdoutRedirector(self.output_text)
        sys.stderr = StdoutRedirector(self.output_text)

    def select_input_path(self):
        path = filedialog.askdirectory()
        if path:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, path)

            # 显示目录内容预览
            preview_window = tk.Toplevel(self.root)
            preview_window.title(f"目录内容预览: {path}")

            # 添加列表框显示文件
            listbox = tk.Listbox(preview_window, width=80, height=20)
            listbox.pack(padx=10, pady=10)

            # 添加滚动条
            scrollbar = tk.Scrollbar(preview_window)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            listbox.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=listbox.yview)

            # 填充目录内容 - 只显示图片文件
            try:
                image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
                files = [f for f in os.listdir(path)
                        if f.lower().endswith(image_extensions)]
                for file in files:
                    listbox.insert(tk.END, file)
            except Exception as e:
                listbox.insert(tk.END, f"无法读取目录内容: {e}")

    def select_output_path(self):
        path = filedialog.askdirectory()
        if path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, path)

    def find_model_file(self):
        # 在可能的路径中查找模型文件
        possible_paths = [
            "model.onnx",
            os.path.join(os.path.dirname(__file__), "model.onnx"),
            os.path.join(os.path.dirname(__file__), "src", "model.onnx"),
            os.path.join(os.path.dirname(__file__), "..", "model.onnx"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def run_inference(self):
        input_path = self.input_entry.get()
        output_path = self.output_entry.get()
        if not input_path:
            messagebox.showerror("错误", "请选择输入路径")
            return

        def run():
            self.is_running = True
            self.run_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.output_text.config(state='normal')
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "开始推理...\n")
            self.output_text.config(state='disabled')

            # No stdout redirection in windowed mode
            def safe_print(msg):
                self.output_queue.append(msg)
                self.root.after(100, self._process_output_queue)

            try:
                self.run_onnx_inference(input_path, output_path, print_func=safe_print)

                if self.is_running:
                    safe_print("\n推理完成！")
                    messagebox.showinfo("完成", "推理完成！")
                else:
                    safe_print("\n推理已中断")
            except Exception as e:
                safe_print(f"\n错误: {str(e)}")
                messagebox.showerror("错误", f"推理过程中出错: {e}")
            finally:
                self.run_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                self.is_running = False

        threading.Thread(target=run, daemon=True).start()

    def _process_output_queue(self):
        if not self.output_queue:
            return
        self.output_text.config(state='normal')
        while self.output_queue:
            msg = self.output_queue.pop(0)
            self.output_text.insert(tk.END, msg)
        self.output_text.see(tk.END)
        self.output_text.config(state='disabled')

    def open_output_dir(self):
        output_path = self.output_entry.get()
        if not os.path.exists(output_path):
            messagebox.showerror("错误", "输出目录不存在")
            return

        try:
            os.startfile(output_path)
        except Exception as e:
            messagebox.showerror("错误", f"无法打开目录: {e}")

    def stop_inference(self):
        if self.is_running:
            self.is_running = False
            self.run_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.output_text.config(state='normal')
            self.output_text.insert(tk.END, "\n推理已手动中断\n")
            self.output_text.see(tk.END)
            self.output_text.config(state='disabled')
            messagebox.showinfo("中断", "推理已中断")

    # 以下是集成的run_onnx.py功能
    def save_img(self, filepath, img):
        try:
            # 使用cv2.imencode处理中文路径
            ext = os.path.splitext(filepath)[1]
            success, buf = cv2.imencode(ext, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if success:
                with open(filepath, 'wb') as f:
                    f.write(buf.tobytes())
            else:
                raise Exception("图片编码失败")
        except Exception as e:
            raise Exception(f"保存图片失败: {filepath}, 错误: {str(e)}")

    def get_inputs(self, data_file_dir):
        inputs = data_file_dir
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif')
        lr = []
        for ext in image_extensions:
            lr.extend(glob.glob(os.path.join(inputs, ext)))
        lr = [{"img": x} for x in sorted(lr)]
        return lr

    def read_img(self, img_path):
        image = np.array(Image.open(img_path).convert('RGB'))
        image = image.astype(np.float32) / 255
        h = image.shape[0]
        w = image.shape[1]
        clean_name = img_path
        image = np.transpose(image, (2, 0, 1))
        return clean_name, image, h, w

    def resize_tensor(self, tensor, size):
        tensor = np.transpose(tensor, (0, 2, 3, 1))
        resized = cv2.resize(tensor[0], (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        if len(tensor.shape) == 4:
            resized = np.expand_dims(resized, axis=0)
        resized = np.transpose(resized, (0, 3, 1, 2))
        return resized

    def process_and_infer(self, x, ort_session):
        _, _, h, w = x.shape
        need_transpose = False
        original_size = (h, w)

        if (h, w) == (1080, 1920):
            x = np.transpose(x, (0, 1, 3, 2))
            need_transpose = True
        elif (h, w) == (1920, 1080):
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
        else:
            if h > w:
                x = self.resize_tensor(x, (1920, 1080))
            else:
                x = np.transpose(x, (0, 1, 3, 2))
                x = self.resize_tensor(x, (1920, 1080))
                need_transpose = True

        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        result = ort_session.run([output_name], {input_name: x})[0]
        result = np.clip(result, 0, 1)

        if need_transpose and original_size == (1080, 1920):
            result = np.transpose(result, (0, 1, 3, 2))
            # 情况2: 之前进行了填充，现在需要裁剪
        elif 'padding' in locals():
            pad_top, pad_bottom, pad_left, pad_right = padding
            # 裁剪掉填充的部分
            result = result[:, :, pad_top:1920 - pad_bottom, pad_left:1080 - pad_right]
        elif original_size not in [(1920, 1080), (1080, 1920)]:
            if need_transpose:
                result = self.resize_tensor(result, (original_size[1], original_size[0]))
                result = np.transpose(result, (0, 1, 3, 2))
            else:
                result = self.resize_tensor(result, original_size)

        result = np.squeeze(np.transpose(result, (0, 2, 3, 1)), axis=0)
        result = np.clip(result, 0, 1)
        result = img_as_ubyte(result)
        return result

    def set_seeds(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        ort.set_seed(seed)

    def run_onnx_inference(self, data_file_dir, output_path, print_func=print):
        self.set_seeds(42)
        inputs = self.get_inputs(data_file_dir)
        save_folder = os.path.join(output_path, "onnx")
        pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
        print_func("save to " + save_folder + '\n')

        onnx_model_path = self.find_model_file()
        if not onnx_model_path:
            raise FileNotFoundError("找不到模型文件 (model.onnx)")

        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(onnx_model_path, sess_options=sess_options)

        # 使用自定义的TqdmGUI
        for img_path in TqdmGUI(inputs, desc="处理图片", gui_text=self.output_text):
            if not self.is_running:
                print("检测到中断信号，停止推理")
                break

            clean_name, degrad_patch, h, w = self.read_img(img_path['img'])
            degrad_patch = np.expand_dims(degrad_patch, axis=0)

            max_retry = 8
            retry_count = 0
            success = False

            while not success and retry_count < max_retry and self.is_running:
                restored = self.process_and_infer(degrad_patch, ort_session)
                restored = cv2.bilateralFilter(restored, d=3, sigmaColor=10, sigmaSpace=10)

                ext = os.path.splitext(clean_name)[-1]
                save_name = os.path.splitext(os.path.split(clean_name)[-1])[0] + ext
                # 确保使用os.path.join和标准化路径
                save_path = os.path.normpath(os.path.join(save_folder, save_name))
                self.save_img(save_path, restored)

                if os.path.getsize(save_path) >= 15 * 1024:
                    success = True
                else:
                    retry_count += 1
                    if retry_count < max_retry and self.is_running:
                        print(
                            f"检测到输出异常({os.path.getsize(save_path) / 1024:.1f}KB)，重试推理 ({retry_count}/{max_retry})")
                    elif not self.is_running:
                        break
                    else:
                        print(f"达到最大重试次数，仍可能存在异常({os.path.getsize(save_path) / 1024:.1f}KB)/{save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDenoisingGUI(root)

    # 等待窗口内容完全加载
    root.update_idletasks()

    # 计算居中位置
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = root.winfo_width()
    window_height = root.winfo_height()

    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # 设置窗口位置
    root.geometry(f"+{x}+{y}")

    root.mainloop()

