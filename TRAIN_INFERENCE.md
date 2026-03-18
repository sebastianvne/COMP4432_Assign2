# Assign2 训练与推理说明

本文档说明如何使用当前 `assign2` 下已经实现好的传统计算机视觉流水线，完成：

- 模型训练
- 检索与分类评估
- 单张图片推理
- 结果文件查看

系统实现对应的主入口文件是 `main.py`，核心流程为：

1. 从 `dataset/dataset.csv` 读取样本
2. 对图像做 `GrabCut` 背景抑制
3. 提取 `HSV(H,S)` 颜色直方图
4. 提取 `SIFT` 局部描述子
5. 用训练集 `SIFT` 描述子训练 `MiniBatchKMeans` 视觉词典
6. 根据关键点到图像中心的距离做线性加权投票，生成空间加权 `BoVW`
7. 将颜色特征与 `BoVW` 拼接成最终 embedding
8. 用 `KNN` 做检索，并以 Top-K 多数投票完成分类

## 目录结构

当前主要文件如下：

- `dataset/dataset.csv`：数据标注文件，至少包含 `path` 和 `label`
- `dataset/*.jpg`：图像数据
- `src/dataset.py`：数据读取、标签映射、分层划分
- `src/preprocess.py`：图像缩放和 `GrabCut`
- `src/features.py`：颜色特征和 `SIFT`
- `src/bovw.py`：视觉词典训练、空间加权 `BoVW`
- `src/train_eval.py`：特征融合、KNN、评估
- `main.py`：训练与推理入口
- `requirements.txt`：依赖列表

## 1. 环境准备

建议在 `assign2` 目录下执行以下命令。

### 1.1 安装依赖

```powershell
python -m pip install -r requirements.txt
```

如果你已经安装过部分依赖，也可以只补装缺失项：

```powershell
python -m pip install joblib numpy opencv-python-headless scikit-learn
```

### 1.2 数据集格式要求

当前代码默认使用 `dataset/dataset.csv` 作为数据入口，格式如下：

```csv
path,label,time,cn_name
./dataset/HJR1.jpg,0,1900,黄金榕
./dataset/JLQ1.jpg,1,1200,金连翘
```

其中：

- `path`：图像相对路径
- `label`：类别编号
- `time`：拍摄时间信息，可选辅助字段，当前实现不参与训练
- `cn_name`：中文类别名，用于日志和结果展示

## 2. 训练流程

### 2.1 最基础训练命令

```powershell
python main.py train
```

这条命令会使用默认参数完成以下工作：

- 从 `dataset/dataset.csv` 读取全部样本
- 按 `label` 做分层划分，默认 `70%` 训练集、`30%` 测试集
- 对训练集和测试集分别提特征
- 仅使用训练集的 `SIFT` 描述子训练视觉词典
- 为所有图像编码加权 `BoVW`
- 融合颜色特征与 `BoVW`
- 建立 `KNN` 检索器
- 输出分类与检索评估结果
- 将模型和中间结果保存到 `outputs/`

### 2.2 推荐的训练命令

如果你想更明确地指定关键参数，推荐使用：

```powershell
python main.py train `
  --csv dataset/dataset.csv `
  --output-dir outputs `
  --num-words 300 `
  --top-k 5 `
  --metric cosine `
  --max-side 960 `
  --test-size 0.3 `
  --random-state 42
```

说明：

- `--num-words 300`：词典大小，与你设计文档中的示例一致
- `--top-k 5`：KNN 检索和投票时取最近的 5 个邻居
- `--metric cosine`：使用余弦距离，适合直方图型特征
- `--max-side 960`：训练前先将图片最长边限制在 960，减小计算量

### 2.3 冒烟测试命令

如果你只是想先快速确认代码能跑通，可以先用更小的词典：

```powershell
python main.py train --num-words 40 --max-side 512 --top-k 3 --output-dir outputs_smoke
```

这适合调试流程，不适合作为最终实验结果。

## 3. 训练阶段的关键步骤说明

### 3.1 数据读取

代码入口位于 `src/dataset.py`。

- 读取 `dataset.csv`
- 将相对路径解析成绝对路径
- 建立 `label -> cn_name` 映射
- 用分层抽样切分训练集和测试集

### 3.2 预处理

代码入口位于 `src/preprocess.py`。

- 先将图像缩放到统一的最大边长度
- 基于“主体位于图像中心”的假设，构造中心矩形
- 使用 `cv2.grabCut` 做前景分割
- 将背景区域置黑，只保留主体区域供后续提特征

### 3.3 颜色特征

代码入口位于 `src/features.py` 中的 `extract_hs_histogram()`。

- 将图像从 `BGR` 转为 `HSV`
- 只统计 `H` 和 `S` 的二维直方图
- 显式丢弃 `V` 通道，降低光照影响
- 最后把直方图归一化

### 3.4 SIFT 特征

代码入口位于 `src/features.py` 中的 `extract_sift_descriptors()`。

- 在灰度图上运行 `SIFT`
- 得到关键点 `keypoints`
- 得到 `128` 维描述子 `descriptors`
- 如果某张图提取不到特征，则返回空描述子，避免训练中断

### 3.5 空间加权 BoVW

代码入口位于 `src/bovw.py`。

主要步骤如下：

1. 将训练集所有 `SIFT` 描述子堆叠
2. 使用 `MiniBatchKMeans` 训练视觉词典
3. 将每个描述子映射到最近的视觉词
4. 不采用普通的 `+1` 投票，而是根据关键点到图像中心的距离做线性衰减加权
5. 将所有视觉词计数形成固定长度直方图
6. 对直方图做 `L1` 归一化

当前默认权重范围：

- 中心区域最高权重：`1.5`
- 边缘区域最低权重：`0.1`

对应命令行参数：

- `--min-spatial-weight`
- `--max-spatial-weight`

### 3.6 特征融合与 KNN

代码入口位于 `src/train_eval.py`。

- 将 `HSV(H,S)` 特征与 `BoVW` 直方图拼接
- 对融合后的向量做 `L2` 归一化
- 构建 `KNN` 检索器
- 使用 Top-K 邻居的多数投票作为最终类别预测

## 4. 训练参数详解

以下是 `python main.py train` 支持的主要参数。

### 数据与划分

- `--csv`：标注文件路径，默认 `dataset/dataset.csv`
- `--output-dir`：输出目录，默认 `outputs`
- `--test-size`：测试集比例，默认 `0.3`
- `--random-state`：随机种子，默认 `42`

### 预处理

- `--max-side`：图像最长边，默认 `960`
- `--grabcut-margin-ratio`：GrabCut 中心矩形边距比例，默认 `0.1`
- `--grabcut-iter-count`：GrabCut 迭代次数，默认 `5`

### 颜色特征

- `--h-bins`：H 通道分箱数，默认 `30`
- `--s-bins`：S 通道分箱数，默认 `32`

### SIFT

- `--sift-nfeatures`：SIFT 最大关键点数，默认 `0`，表示 OpenCV 自行决定
- `--sift-contrast-threshold`：低对比度过滤阈值，默认 `0.04`
- `--sift-edge-threshold`：边缘响应阈值，默认 `10.0`
- `--sift-sigma`：高斯尺度参数，默认 `1.6`

### BoVW

- `--num-words`：视觉词典大小，默认 `300`
- `--kmeans-batch-size`：MiniBatchKMeans 批大小，默认 `2048`
- `--kmeans-max-iter`：KMeans 最大迭代次数，默认 `100`
- `--min-spatial-weight`：边缘最小权重，默认 `0.1`
- `--max-spatial-weight`：中心最大权重，默认 `1.5`

### 特征融合与检索

- `--color-weight`：颜色特征权重，默认 `1.0`
- `--bovw-weight`：BoVW 特征权重，默认 `1.0`
- `--metric`：KNN 距离度量，可选 `cosine` 或 `euclidean`
- `--top-k`：检索邻居数，默认 `5`

## 5. 训练输出文件说明

训练完成后，输出目录中通常会包含以下文件：

- `model_bundle.joblib`
- `vocab.joblib`
- `train_features.npy`
- `train_bovw_features.npy`
- `train_color_features.npy`
- `train_labels.npy`
- `test_features.npy`
- `test_labels.npy`
- `label_map.json`
- `metrics.json`
- `sample_paths.json`
- `retrieval_examples.json`

### 5.1 `model_bundle.joblib`

这是推理阶段最重要的文件，内部包含：

- 训练好的视觉词典
- 训练集 embedding
- 训练集标签
- 训练集样本路径
- 标签名映射
- 训练配置参数

推理时默认直接加载这个文件。

### 5.2 `metrics.json`

包含训练后评估指标，例如：

- `classification.accuracy`
- `classification.macro_f1`
- `classification.confusion_matrix`
- `retrieval.top_1_accuracy`
- `retrieval.top_5_accuracy` 或当前实际使用的 `top_k`
- `retrieval.mAP`

### 5.3 `retrieval_examples.json`

保存测试集样本的检索结果示例，便于：

- 查看 query 的预测类别
- 查看 Top-K 邻居
- 做实验分析
- 写报告时展示案例

## 6. 如何做单张图片推理

### 6.1 基础命令

```powershell
python main.py predict --image dataset\HJR1.jpg
```

默认会加载：

```text
outputs/model_bundle.joblib
```

如果你的模型保存在别的目录，比如 `outputs_smoke`，则这样写：

```powershell
python main.py predict --image dataset\HJR1.jpg --model outputs_smoke\model_bundle.joblib --top-k 3
```

### 6.2 推理时系统内部做了什么

给定一张新图，推理阶段会执行：

1. 读取图片
2. 按训练阶段相同参数做 `GrabCut`
3. 提取 `HSV(H,S)` 直方图
4. 提取 `SIFT` 描述子
5. 使用训练好的视觉词典生成加权 `BoVW`
6. 将颜色特征与 `BoVW` 融合为最终向量
7. 与训练集特征做 KNN 检索
8. 返回 Top-K 邻居及多数投票结果

### 6.3 推理输出示例

输出是一个 JSON 结构，形如：

```json
{
  "predicted_label": 0,
  "predicted_name": "黄金榕",
  "neighbors": [
    {
      "path": "./dataset/HJR1.jpg",
      "label": 0,
      "cn_name": "黄金榕",
      "score": 0.99
    }
  ]
}
```

字段含义：

- `predicted_label`：预测类别编号
- `predicted_name`：预测类别名
- `neighbors`：最近邻检索结果
- `score`：相似度分数；使用 `cosine` 时显示为 `1 - distance`

## 7. 推荐实验流程

如果你要做正式实验，建议按下面顺序进行。

### 第一步：先做冒烟测试

```powershell
python main.py train --num-words 40 --max-side 512 --top-k 3 --output-dir outputs_smoke
```

目的：

- 检查环境是否正常
- 检查 `cv2`、`sklearn` 是否安装成功
- 检查训练、评估、保存流程是否完整

### 第二步：跑正式训练

```powershell
python main.py train --num-words 300 --top-k 5 --metric cosine --output-dir outputs
```

### 第三步：查看评估结果

重点查看：

- `outputs/metrics.json`
- `outputs/retrieval_examples.json`

### 第四步：跑单图推理

```powershell
python main.py predict --image dataset\JLQ1.jpg --model outputs\model_bundle.joblib --top-k 5
```

## 8. 常见问题

### 8.1 报错 `No module named 'cv2'`

说明没有安装 OpenCV，执行：

```powershell
python -m pip install opencv-python-headless
```

### 8.2 某些图片提取不到 SIFT 特征怎么办

当前代码已经做了兜底：

- 若没有关键点，则返回空描述子
- 编码阶段会输出零向量，不会直接报错中断

但这会影响识别效果，所以你可以：

- 调整 `--sift-contrast-threshold`
- 调整 `--max-side`
- 检查 `GrabCut` 是否把主体裁掉太多

### 8.3 词典大小 `K` 应该怎么选

建议：

- 调试阶段：`40` 到 `100`
- 正式实验：`200` 到 `300`

样本量较小时，词典过大容易导致：

- `BoVW` 稀疏
- 检索不稳定
- 训练时间变长

### 8.4 为什么建议优先用 `cosine`

因为：

- 颜色直方图和 `BoVW` 本质上都是分布型特征
- 当前特征经过归一化后，用余弦距离通常比欧氏距离更稳

## 9. 一份可直接复制的完整命令

### 安装依赖

```powershell
python -m pip install -r requirements.txt
```

### 训练

```powershell
python main.py train --csv dataset/dataset.csv --output-dir outputs --num-words 300 --top-k 5 --metric cosine
```

### 推理

```powershell
python main.py predict --image dataset\HJR1.jpg --model outputs\model_bundle.joblib --top-k 5
```

## 10. 报告撰写建议

如果你要把这个系统写进课程报告，建议按下面的结构描述实验过程：

1. 数据集说明：类别数、样本数、拍摄条件
2. 预处理：中心先验 `GrabCut`
3. 特征设计：`HSV(H,S)` + `SIFT`
4. 编码方法：空间加权 `BoVW`
5. 分类方式：`KNN + Majority Voting`
6. 指标：`accuracy`、`macro F1`、`Top-k`、`mAP`
7. 结果分析：成功案例、失败案例、参数影响

这样就能和你当前实现一一对应。
