import torch
import torch.nn.functional as F
import numpy as np

class InverseUVMapGenerator:
    fill_methods = ["none", "sparse"]  # Define fill methods: no fill or sparse fill

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uvmap": ("IMAGE",),  # Input UV map [1, H, W, 3]
                "fill_method": (cls.fill_methods, {"default": "none"}),  # Fill method options
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "generate_inverse_uvmap"
    CATEGORY = "FunctionAsTexture"

    def generate_inverse_uvmap(self, uvmap, fill_method="none"):
        batch, height, width, _ = uvmap.shape

        # Initialize the inverse UV map with default values: [1, 1, 0]
        inverse_uvmap = torch.zeros((1, height, width, 3), device=uvmap.device)
        inverse_uvmap[..., :2] = 1  # Set R and G channels to default value 1

        # Extract the R and G channels (U and V coordinates) from the input UV map
        u_channel = (uvmap[0, :, :, 0] * (width - 1)).round().long()
        v_channel = (uvmap[0, :, :, 1] * (height - 1)).round().long()

        # Clamp the coordinates to ensure they are within valid ranges
        u_channel = torch.clamp(u_channel, 0, width - 1)
        v_channel = torch.clamp(v_channel, 0, height - 1)

        # Fill the inverse UV map based on the chosen fill method
        if fill_method == "none":
            inverse_uvmap = self.fill_inverse_uvmap_no_fill(u_channel, v_channel, inverse_uvmap, width, height)
        elif fill_method == "sparse":
            inverse_uvmap = self.fill_inverse_uvmap_sparse(u_channel, v_channel, inverse_uvmap, width, height)
        # Generate the mask as a separate tensor
        mask = np.ceil(inverse_uvmap[:, :, :, 0])
        return (inverse_uvmap.float(), mask.float())

    def fill_inverse_uvmap_no_fill(self, u_channel, v_channel, inverse_uvmap, width, height):
        # No filling method, just write values
        for y in range(height):
            for x in range(width):
                tx, ty = u_channel[y, x].item(), v_channel[y, x].item()
                inverse_uvmap[0, ty, tx, 0] = x / (width - 1)  # Write U value to the R channel
                inverse_uvmap[0, ty, tx, 1] = y / (height - 1)  # Write V value to the G channel

        # Compute multiplication result: 1 - floor(UVmap's R channel)
        r_channel_floor = torch.floor(inverse_uvmap[:, :, :, 0])
        inverse_uvmap *= (1 - r_channel_floor.unsqueeze(-1))  # Apply the multiplication to all channels

        return inverse_uvmap

    def fill_inverse_uvmap_sparse(self, u_channel, v_channel, inverse_uvmap, width, height):
        # Fill the inverse UV map with sparse filling
        for y in range(height):
            for x in range(width):
                tx, ty = u_channel[y, x].item(), v_channel[y, x].item()
                inverse_uvmap[0, ty, tx, 0] = x / (width - 1)  # Write U value to the R channel
                inverse_uvmap[0, ty, tx, 1] = y / (height - 1)  # Write V value to the G channel

                # Fill the adjacent pixels to reduce holes
                if ty + 1 < height:
                    inverse_uvmap[0, ty + 1, tx, 0] = x / (width - 1)
                    inverse_uvmap[0, ty + 1, tx, 1] = y / (height - 1)
                if tx + 1 < width:
                    inverse_uvmap[0, ty, tx + 1, 0] = x / (width - 1)
                    inverse_uvmap[0, ty, tx + 1, 1] = y / (height - 1)

                    inverse_uvmap[0, ty + 1, tx + 1, 0] = x / (width - 1)
                    inverse_uvmap[0, ty + 1, tx + 1, 1] = y / (height - 1)

        # Apply sparse fill if selected
        inverse_uvmap = self.nearest_fill(inverse_uvmap)

        # Ensure B channel is always 0
        inverse_uvmap[:, :, :, 2] = 0

        # Compute multiplication result: 1 - floor(UVmap's R channel)
        r_channel_floor = torch.floor(inverse_uvmap[:, :, :, 0])
        inverse_uvmap *= (1 - r_channel_floor.unsqueeze(-1))  # Apply multiplication to all channels

        return inverse_uvmap

    def nearest_fill(self, tensor):
        # Sparse fill to handle unassigned regions, using nearest neighbor values
        filled = tensor.clone()
        for c in range(filled.shape[-1] - 1):  # Fill R and G channels, ignore B channel
            mask = (tensor[..., c] == 0)  # Identify unassigned regions
            while mask.any():
                # Propagate non-zero pixel values using a 3x3 convolution kernel
                filled_channel = F.conv2d(
                    filled[:, :, :, c].unsqueeze(1),  # Extract the current channel [batch, 1, H, W]
                    weight=torch.ones(1, 1, 3, 3, device=tensor.device),  # Convolution kernel
                    padding=1
                )
                filled[:, :, :, c][mask] = filled_channel.squeeze(1)[mask]  # Write the filled values back
                mask = (filled[..., c] == 0)  # Update the mask
        return filled

class TextureSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texture": ("IMAGE",),
                "uvmap": ("IMAGE",),
                "interpolation": (["bilinear", "bicubic"], {"default": "bicubic"}),  # 新增插值方式选择
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remap_texture"
    CATEGORY = "FunctionAsTexture"

    def remap_texture(self, texture, uvmap, interpolation="bicubic"):
        # 检查输入张量形状
        if len(texture.shape) != 4 or texture.shape[-1] != 3:
            raise ValueError("Texture must be a four-dimensional tensor with shape [batch, H, W, 3].")
        
        if len(uvmap.shape) != 4 or uvmap.shape[0] != 1 or uvmap.shape[-1] != 3:
            raise ValueError("UV map must be a four-dimensional tensor with shape [1, H, W, 3].")

        # 获取 UV 映射的高度和宽度
        uv_height, uv_width = uvmap.shape[1:3]

        # 调整纹理尺寸
        if texture.shape[1:3] != (uv_height, uv_width):
            texture_resized = F.interpolate(
                texture.permute(0, 3, 1, 2),
                size=(uv_height, uv_width),
                mode=interpolation,  # 使用选择的插值方式
                align_corners=True
            ).permute(0, 2, 3, 1)
        else:
            texture_resized = texture
       
        # 转换为 grid_sample 格式
        texture_permuted = texture_resized.permute(0, 3, 1, 2)

        # 处理 UV 坐标
        uvmap_2d = uvmap[:, :, :, :2].float()
        # 使用更高精度的中间计算
        normalized_uvmap = ((uvmap_2d * 2).to(torch.float32) - 1).to(torch.float32)
        
        # 扩展 UV 映射的 batch 维度
        normalized_uvmap_expanded = normalized_uvmap.expand(texture.shape[0], -1, -1, -1)

        # 使用更高精度的 grid_sample
        remapped_texture = F.grid_sample(
            texture_permuted,
            normalized_uvmap_expanded,
            mode=interpolation,  # 使用选择的插值方式
            padding_mode='border',
            align_corners=True
        )

        # 转回原始格式
        remapped_texture_final = remapped_texture.permute(0, 2, 3, 1)

        return (remapped_texture_final.float(),)

class UVCoordinateGen:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 256, "min": 1, "max": 16384, "step": 1}),
                "height": ("INT", {"default": 256, "min": 1, "max": 16384, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_uv_coordinates"
    CATEGORY = "FunctionAsTexture"

    def generate_uv_coordinates(self, width, height, scale):
        y, x = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing='ij')
        u = torch.fmod(x * scale, 1)  
        v = torch.fmod(y * scale, 1)     
        
        uv_tensor = torch.stack([u, v, torch.zeros_like(u)], dim=-1)  
        uv_tensor = uv_tensor.unsqueeze(0)  

        return (uv_tensor.float(),)



NODE_CLASS_MAPPINGS = {
    "UVCoordinateGen": UVCoordinateGen,
    "TextureSampler": TextureSampler,
    "InverseUVMapGenerator": InverseUVMapGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UVCoordinateGen": "UV Coordinate Generator",
    "TextureSampler": "Texture Sampler",
    "InverseUVMapGenerator": "Inverse UV Map Generator",
}