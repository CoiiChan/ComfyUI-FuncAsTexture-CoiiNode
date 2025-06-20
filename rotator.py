import torch
import torch.nn.functional as F
import numpy as np

class Rotator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入的图像
                "rotation": ("FLOAT", {"default": 45.0, "min": -360, "max": 360, "step": 0.01}),  # 旋转值
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),  # 旋转中心 x 坐标
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number"}),  # 旋转中心 y 坐标
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rotate_image"
    CATEGORY = "FunctionAsTexture"

    def rotate_image(self, image, rotation, center_x, center_y):
        # 确保输入图像是一个 [batch, H, W, 3] 张量
        if len(image.shape) != 4 or image.shape[-1] != 3:
            raise ValueError("Input image must have shape [batch, H, W, 3].")

        # 获取图像的高度和宽度
        batch_size, height, width, channels = image.shape

        # 计算旋转角度：rotation 范围从 0 到 1，对应 0 到 360 度,由于comfyui精度仅到0.01，因此不建议使用angle缩放
        #angle = rotation * 360.0

        # 计算旋转矩阵的中心坐标 # 将中心坐标归一化UV画布的[0,0,1,1]→[-1,-1,1,1]
        center_x_pixel = (center_x * 2) -1
        center_y_pixel = (center_y * 2) -1

        # 构建旋转矩阵
        theta = self._get_rotation_matrix(center_x_pixel, center_y_pixel, rotation)

        # 将图像的维度转换为 [batch, 3, H, W]，适用于 grid_sample
        image_permuted = image.permute(0, 3, 1, 2)  # 变为 [batch, 3, H, W]

        # 创建网格
        grid = F.affine_grid(theta, image_permuted.size(), align_corners=False)

        # 使用 grid_sample 执行旋转
        rotated_image = F.grid_sample(image_permuted, grid, align_corners=False)

        # 将结果转换回 [batch, H, W, 3] 格式
        rotated_image = rotated_image.permute(0, 2, 3, 1)

        return (rotated_image.float(),)

    def _get_rotation_matrix(self, cx, cy, angle):
        # """
        # 生成旋转矩阵
        # :param cx: 旋转中心 x 坐标
        # :param cy: 旋转中心 y 坐标
        # :param angle: 旋转角度 (degrees)
        # :return: 旋转矩阵
        # """
        # 转换为弧度
        angle_rad = np.radians(angle)

        # 计算旋转矩阵
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)

        # 旋转矩阵公式
        matrix = np.array([[cos_theta, -sin_theta, cx - cx * cos_theta + cy * sin_theta],
                           [sin_theta, cos_theta, cy - cx * sin_theta - cy * cos_theta]])

        # 将其转换为 PyTorch 张量，并确保维度为 [1, 2, 3]
        return torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)  # 形状 [1, 2, 3]
        

# ComfyUI 内注册节点
NODE_CLASS_MAPPINGS = {
    "Rotator": Rotator
}

# 设置节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "Rotator": "Rotator"
}
