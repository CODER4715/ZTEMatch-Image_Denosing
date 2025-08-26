# 基础镜像：使用 NVIDIA CUDA 基础镜像（包含 CUDA 工具链和驱动）
FROM ubuntu:22.04

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    vim \
    ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 更新pip
RUN python3.10 -m pip install --no-cache-dir --upgrade pip

# 复制项目文件（根据实际情况调整）
COPY . .

# 安装 Python 依赖（根据项目需求调整）

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lightning==2.0.1
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ptflops==0.7.4
RUN pip cache purge


# 设置环境变量（可选）
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# 默认命令（启动脚本或服务）
# CMD ["python", "src/inference.py"]
ENTRYPOINT ["python3", "src/inference.py"]

#docker run --gpus all -it --rm -v D:\\1_Image_Denoising\\ZTE_Data:/app/data moeir --data_file_dir /app/data