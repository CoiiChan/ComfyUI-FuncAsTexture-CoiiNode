# CustomScriptNumpy 节点使用指南
[English](https://github.com/CoiiChan/ComfyUI-FuncAsTexture-CoiiNode)|[中文](https://github.com/CoiiChan/ComfyUI-FuncAsTexture-CoiiNode/blob/main/README_CN.md)
## 概览
通过NumPy公式对输入图像进行数学运算和通道进行自由更自由精准的操作，适合有编程经验者的Comfyui使用者。

![overview](https://github.com/CoiiChan/ComfyUI-FuncAsTexture-CoiiNode/blob/main/example/Overview.png?raw=true)

我希望基础的image(s)计算权重不再靠猜和试而是有清晰的可视的规则。

因为时常无法分辨混合模式normal、screen、overlay 分别是什么权重组合，甚至难以找到一个Max（image a，image b）的操作。

主节点灵感来源虚幻引擎材质编辑器的Custom节点

![inspiredfromUE5](https://github.com/CoiiChan/ComfyUI-FuncAsTexture-CoiiNode/blob/main/example/inspiredfromUE5.png?raw=true)

## 核心功能
## 主节点输入参数说明

| 参数名 | 必填 | 描述 |
|--------|------|------|
| imga   | 否   | 主输入图像，自动分解为imga_r, imga_g, imga_b三个通道，作为输出尺寸的参考 |
| imgb   | 否   | 可选输入图像，未提供时使用全零数组 |
| imgc   | 否   | 可选输入图像，未提供时使用全零数组 |
| pfloat1~3  | 否   | 可选浮点型参数，可传递负数，默认值为 0.0，用于脚本计算 |
| pint1~2  | 否   | 可选整型参数，可传递负数，默认值为 0，用于脚本计算 |
| formula| 是   | 多行NumPy运算公式，用于处理图像数据 |

![mainnode](https://github.com/CoiiChan/ComfyUI-FuncAsTexture-CoiiNode/blob/main/example/mainnode.png?raw=true)

## 注意事项
1. 默认已经包含import numpy as np ，在文本中可以省略，使用赋值result输出 
2. 输出会自动转换为4D张量格式[:,H,W,C],输出H,W 高宽默认与imga输入相同，缺省时则为512,512
3. 多行文本formula可以使用NumPy的所有数学函数(np.sin, np.exp，矩阵计算和条件语句)
4. 输入的多图计算时，注意它们的分辨率是否需要相同，避免报错

## 派生节点
- 作为主节点的派生的子节即是使用示例也可直接使用和直接修改function脚本。
- 派生节点移除了节点本身不需要的输入项和参数的输入范围以使其更符合直接使用习惯。
- 作为学习numpy计算的学习者可以将派生节点function多行文本复制到主节点中去验证和自主扩展。

派生节点包含：

| 节点名 | 描述 |
| --- | --- |
| Subtraction | 对两个可选输入图像进行减法运算，结果赋值给 `result`。 |
| Multiply | 对两个可选输入图像进行乘法运算，结果赋值给 `result`。 |
| Divided | 对两个可选输入图像进行除法运算，结果赋值给 `result`。 |
| Max | 取两个可选输入图像对应位置的最大值，结果赋值给 `result`。 |
| Min | 取两个可选输入图像对应位置的最小值，结果赋值给 `result`。 |
| Dot | 计算两个可选输入图像在最后一个轴上的点积，结果赋值给 `result`。 |
| Distance | 计算两个可选输入图像对应位置的欧几里得距离，结果赋值给 `result`。 |
| Power | 对可选输入图像 `imga` 进行幂运算，指数由 `pfloat1` 指定，结果赋值给 `result`。 |
| HueShift | 对可选输入图像 `imga` 进行色相偏移，偏移角度由 `pfloat1` 指定，结果赋值给 `result`。 |
| Panner | 对可选输入图像 `imga` 进行平移操作，x 和 y 方向的偏移量分别由 `pint1` 和 `pint2` 指定，结果赋值给 `result`。 |
| Outline | 实现四向偏移卷积，通过计算图像左右上下各偏移 `pint1` 像素后的图像，求和取平均再减去原始图像，以获取图像轮廓。 |
| Lerp | 根据浮点数 `pfloat1` 对两个可选输入图像 `imga` 和 `imgb` 进行线性插值，结果赋值给 `result`。 |
| Clamp | 将可选输入图像 `imga` 的像素值限制在 `pfloat1` 和 `pfloat2` 指定的范围内，结果赋值给 `result`。 |
| Ceil | 对可选输入图像 `imga` 的每个像素值进行向上取整操作，结果赋值给 `result`。 |
| 1-x | 对可选输入图像 `imga` 的每个像素值执行 1 减去该值的操作，结果赋值给 `result`。 |
| Sine | 对可选输入图像 `imga` 的每个像素值计算正弦值，结果赋值给 `result`。 |
| DDX | 对可选输入图像 `imga` 进行 x 方向的差分运算，结果赋值给 `result`。 |
| Contant3Vector(Color) | 使用常量创建RGB纯色图像。 |
| if (FuncAsTexture) | 根据 `imga` 与常量 `pfloat1` 的比较结果，选择 `imgb` 或 `imgc` 作为输出，结果赋值给 `result`。 |
| Chroma_Key_Alpha | 实现绿幕抠像功能，将图像从 RGB 转换为 HSV 颜色空间，定义绿色范围，创建并羽化遮罩，最后将遮罩应用到图像。 |
| Desaturation | 对图像`imga`三通道均值方式去饱和度。 |


## Texture扩展节点

| 节点名 | 描述 |
| --- | --- |
| UV Coordinate Generator | 创建UV纹理 |
| Texture Sampler | 使用UV作为坐标引导的采样变换 |
| Inverse UV Map Generator | 输入UV图求解其逆转变UV图，测试版，像素不宜过大 |
| Rotator | 旋转贴图 |

![extensionnode](https://github.com/CoiiChan/ComfyUI-FuncAsTexture-CoiiNode/blob/main/example/extension.png?raw=true)

## 更多参考链接
- [NumPy数学函数](https://numpy.org/doc/stable/reference/routines.math.html)
- [NumPy统计函数](https://numpy.org/doc/stable/reference/routines.statistics.html)
- [NumPy线性代数](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [图像处理技巧](https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html)

## 基础运算示例
### 1. 图像混合运算
```python
# 简单相加
result = (imga + imgb) * 0.5
```
### 2. 通道操作
```python
# 因使用频繁，初始已经定义了makergb(r, g, b)函数来组合三通道，和定义了imga_r、imga_g、imga_b参数为imaga的r，g，b通道值的复制
#Default already import numpy as np
imga_r = (imga_r + imga_g) * 0.5
imga = makergb(imga_r, imga_g, imga_b)
result = imga

# 加权混合
result = imga*0.7 + imgb*0.3

# 二值化处理
threshold = 0.5
result = np.where(imga > threshold, 1.0, 0.0)# RGB转灰度
gray = 0.299 * imga_r + 0.587 * imga_g + 0.114 * imga_b
result = makergb(gray, gray, gray)
```
## 高级应用示例
### 3. 图像处理效果
```python
# 边缘检测 (3x3 Sobel算子)
kernel_x = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
kernel_y = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])
edge = np.sqrt(np.sum(kernel_x * imga[1:-1,1:-1])**2 + 
              np.sum(kernel_y * imga[1:-1,1:-1])**2)
result = makergb(edge, edge, edge)
```
### 4.特效处理
```python
# 添加随机噪声
noise = np.random.random(imga.shape) * 0.1
result = np.clip(imga + noise, 0, 1)

# 绿幕抠像获得Alpha
#Choma threshold: pfloat1
#Soft Blend: pfloat2
imgb =imgc + imgb[0, 1, 1, :]
chomachannel = np.sum((imga * imgb), axis=-1)
softmask = chomachannel - np.sum(imga , axis=-1)/3
result = 1- np.ceil(softmask - pfloat1) - softmask * pfloat2
```

---
[![CoiiChan](https://avatars.githubusercontent.com/u/49615294?v=4)](https://github.com/CoiiChan)

作者不会制作大一统的节点全家桶，因为多种应用场合的节点有依赖要求，庞大的工具将会在安装上存在很大不便利。

点个小星星关注下，一个自制发散装专项custom node的i人

