---
license: MIT License
image:
  image-editing:
    size_scale:
      - 100-10k
tags: [NTIRE 2020]

---
## 数据集描述
### 数据集简介
智能手机图像去噪训练数据集（SIDD）及NTIRE 2020 Real Image Denoising Challenge提供的验证图片集SIDD+。

数据集包含了成对的噪声图像和干净图像。
### 数据集原始地址
SIDD训练数据集（http://www.cs.yorku.ca/~kamel/sidd/dataset.php)
SIDD-Medium Dataset sRGB images only（~12G）

SIDD验证数据集（https://www.eecs.yorku.ca/~kamel/sidd/benchmark.php）

SIDD+验证数据集（https://competitions.codalab.org/competitions/22231）

### 数据集支持的任务
手机图像去噪
## 数据集的格式和结构
### 数据格式
PNG
### 数据集加载方式
```python
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
ms_ds_valid = MsDataset.load(
            'SIDD', namespace='huizheng', subset_name='default', split='validation', download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
print(next(iter(ms_ds_valid)))

ms_ds_test = MsDataset.load(
            'SIDD', namespace='huizheng', subset_name='default', split='test', download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
print(next(iter(ms_ds_test)))
```
### 数据分片

| 子数据集    |        train | validation |     test |
|---------|-------------:|-----------:|---------:|
| default |   原始SIDD训练数据 |  SIDD+ 验证集 | SIDD 验证集 |
| crops   | 裁剪后的SIDD训练数据 |          / |        / |

## 数据集版权信息
The dataset is under the MIT License.

### Clone with HTTP
* http://www.modelscope.cn/datasets/huizheng/SIDD.git