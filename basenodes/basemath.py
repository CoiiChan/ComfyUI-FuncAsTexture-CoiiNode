
from PIL import Image
import torch
import numpy as np
from ..custom_script_numpy import CustomScript

RETURN_TYPES = ("IMAGE",)
FUNCTION = "execute"
CATEGORY = "FunctionAsTexture/basic"


#Input 3 images Nodes
class ifFunction(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { "default": "result = imga + imgb", "multiline": True }),
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "imgb": ("IMAGE", {"tooltip": "Defined OutPut1,When imga >= constant pfloat1"}), 
                "imgc": ("IMAGE", {"tooltip": "Defined OutPut2,When imga < constant pfloat1"}), 
                "pfloat1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "imga compare with constant pfloat1."}), 
            },
        }
        # 隐藏  pfloat2, pfloat3, pint1, pint2
        for param in [ 'pfloat2', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default": "#if function info: if imga < constant pfloat1: use imgc else: use imgb\n#equalthreshold = 0.01\n#lessweight = np.floor(1- imga + pfloat1- equalthreshold)\nlessweight = np.floor(1- imga + pfloat1)\nlargeweight = np.ceil(imga - pfloat1)\nresult = imgb*largeweight + imgc*lessweight", "multiline": True})
        return inputs

#Input 2 images Nodes
class Add(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { "default": "result = imga + imgb", "multiline": True }),
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "imgb": ("IMAGE", {"tooltip": "Optional"}), 
            },
        }
        # 隐藏 imgc, pfloat1, pfloat2, pfloat3, pint1, pint2
        for param in ['imgc', 'pfloat1', 'pfloat2', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default": "result = imga + imgb", "multiline": True})
        return inputs

class Subtraction(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { "default": "result = imga - imgb", "multiline": True }),
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "imgb": ("IMAGE", {"tooltip": "Optional"}), 
            },
        }
        # 隐藏 imgc, pfloat1, pfloat2, pfloat3, pint1, pint2
        for param in ['imgc', 'pfloat1', 'pfloat2', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default": "result = imga - imgb", "multiline": True})
        return inputs

class Multiply(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { "default": "result = imga * imgb", "multiline": True }),
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "imgb": ("IMAGE", {"tooltip": "Optional"}), 
            },
        }
        # 隐藏 imgc, pfloat1, pfloat2, pfloat3, pint1, pint2
        for param in ['imgc', 'pfloat1', 'pfloat2', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default": "result = imga * imgb", "multiline": True})
        return inputs

class Divided(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { "default": "result = imga / imgb", "multiline": True }),
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "imgb": ("IMAGE", {"tooltip": "Optional"}), 
            },
        }
        # 隐藏 imgc, pfloat1, pfloat2, pfloat3, pint1, pint2
        for param in ['imgc', 'pfloat1', 'pfloat2', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default": "result = imga / imgb", "multiline": True})
        return inputs

class Max(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { "default": "result = np.maximum(imga, imgb)", "multiline": True }),
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "imgb": ("IMAGE", {"tooltip": "Optional"}), 
            },
        }
        # 隐藏 imgc, pfloat1, pfloat2, pfloat3, pint1, pint2
        for param in ['imgc', 'pfloat1', 'pfloat2', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default": "result = np.maximum(imga, imgb)", "multiline": True })
        return inputs


class Min(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { "default": "result = np.minimum(imga, imgb)", "multiline": True }),
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "imgb": ("IMAGE", {"tooltip": "Optional"}), 
            },
        }
        # 隐藏 imgc, pfloat1, pfloat2, pfloat3, pint1, pint2
        for param in ['imgc', 'pfloat1', 'pfloat2', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default": "result = np.minimum(imga, imgb)", "multiline": True })
        return inputs



class Dot(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { 
                    "default": " result = np.sum((imga * imgb), axis=-1)", 
                    "multiline": True 
                }), 
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "imgb": ("IMAGE", {"tooltip": "Optional"}), 
            },
        }
        # 隐藏 imgc, pfloat1, pfloat2, pfloat3, pint1, pint2
        for param in ['imgc', 'pfloat1', 'pfloat2', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})

        inputs["required"]["formula"] = ("STRING", {"default": "result = np.sum((imga * imgb), axis=-1)",  "multiline": True })
        return inputs



class Distance(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { 
                    "default": "result = np.sqrt(np.sum((imga - imgb) ** 2, axis=-1))", 
                    "multiline": True 
                }), 
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "imgb": ("IMAGE", {"tooltip": "Optional"}), 
            },
        }
        # 隐藏 imgc, pfloat1, pfloat2, pfloat3, pint1, pint2
        for param in ['imgc', 'pfloat1', 'pfloat2', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default":"result = np.sqrt(np.sum((imga - imgb) ** 2, axis=-1))", "multiline": True})
        return inputs


class Lerp(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { 
                    "default": "result = imga * pfloat1 + imgb * (1 - pfloat1)", 
                    "multiline": True 
                }), 
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "imgb": ("IMAGE", {"tooltip": "Optional"}), 
                "pfloat1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Optional,Floating-point parameter passed to the script."}), 
            },
        }
        # 隐藏 imgc, pfloat2, pfloat3, pint1, pint2
        for param in ['imgc', 'pfloat2', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default":"result = imgb * pfloat1 + imga * (1 - pfloat1)", "multiline": True})
        return inputs

#Input 1 image 2 INT Nodes
class Ceil(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { 
                    "default": "result = imga * pfloat1 + imgb * (1 - pfloat1)", 
                    "multiline": True 
                }), 
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
            },
        }
        # 隐藏 imgc, pfloat1,pfloat2,pfloat3, pint1, pint2
        for param in ['imgb','imgc', 'pfloat3','pfloat1','pfloat2', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default":"result = np.ceil(imga)", "multiline": True})
        return inputs

class Oneminus(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { 
                    "default": "result = imga * pfloat1 + imgb * (1 - pfloat1)", 
                    "multiline": True 
                }), 
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
            },
        }
        # 隐藏 imgc, pfloat1,pfloat2,pfloat3, pint1, pint2
        for param in ['imgb','imgc', 'pfloat3','pfloat1','pfloat2', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default":"result = 1 - imga", "multiline": True})
        return inputs

class Sine(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { 
                    "default": "result = imga * pfloat1 + imgb * (1 - pfloat1)", 
                    "multiline": True 
                }), 
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
            },
        }
        # 隐藏 imgc, pfloat1,pfloat2,pfloat3, pint1, pint2
        for param in ['imgb','imgc', 'pfloat3','pfloat1','pfloat2', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default":"result = np.sin(imga)", "multiline": True})
        return inputs

class DDX(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", {
                    "default": "x_shift = pint1\ny_shift = pint2\nheight, width = imga.shape[1:3]\nnew_x = np.clip(np.arange(width) + x_shift, 0, width - 1)\nnew_y = np.clip(np.arange(height) + y_shift, 0, height - 1)\nX, Y = np.meshgrid(new_x, new_y)\nresult = imga[:, Y, X, :]",
                    "multiline": True
                }),
            },
            "optional": {
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}),
            },
        }
        # 隐藏 imgb, imgc, pfloat1, pfloat2, pfloat3, pint1, pint2
        for param in ['imgb', 'imgc', 'pfloat1', 'pfloat2', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default":"x_shift = 1\ny_shift = 0\nheight, width = imga.shape[1:3]\nnew_x = np.clip(np.arange(width) + x_shift, 0, width - 1)\nnew_y = np.clip(np.arange(height) + y_shift, 0, height - 1)\nX, Y = np.meshgrid(new_x, new_y)\nresult = (imga[:, Y, X, :]-imga)/16 *width", "multiline": True})
        return inputs


class Clamp(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { 
                    "default": "result = imga * pfloat1 + imgb * (1 - pfloat1)", 
                    "multiline": True 
                }), 
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "pfloat1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Optional,Clamp Min."}), 
                "pfloat2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Optional,Clamp Max."}), 
            },
        }
        # 隐藏 imgc, pfloat3, pint1, pint2
        for param in ['imgb','imgc', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default":"result = np.clip(imga, pfloat1, pfloat2)", "multiline": True})
        return inputs


class Power(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { 
                    "default": "result = imga ** pfloat1", 
                    "multiline": True 
                }), 
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "pfloat1": ("FLOAT", {"default": 2.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "Optional,Floating-point parameter passed to the script."}), 
            },
        }
        # 隐藏 imgb, imgc, pfloat2, pfloat3, pint1, pint2
        for param in ['imgb', 'imgc', 'pfloat2', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default":"result = imga ** pfloat1", "multiline": True})
        return inputs


class HueShift(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { 
                    "default": "result = imga", 
                    "multiline": True 
                }), 
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "pfloat1": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Optional,Floating-point parameter passed to the script."}), 
            },
        }
        # 隐藏 imgb, imgc, pfloat2, pfloat3, pint1, pint2
        for param in ['imgb', 'imgc', 'pfloat2', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {
            "default": "angle = pfloat1 * 2 * np.pi\ncos_theta = np.cos(angle)\nsin_theta = np.sin(angle)\nrotation_matrix = np.array([\n    [cos_theta + (1 - cos_theta) / 3, (1 - cos_theta) / 3 - np.sqrt(3) * sin_theta / 3, (1 - cos_theta) / 3 + np.sqrt(3) * sin_theta / 3],\n    [(1 - cos_theta) / 3 + np.sqrt(3) * sin_theta / 3, cos_theta + (1 - cos_theta) / 3, (1 - cos_theta) / 3 - np.sqrt(3) * sin_theta / 3],\n    [(1 - cos_theta) / 3 - np.sqrt(3) * sin_theta / 3, (1 - cos_theta) / 3 + np.sqrt(3) * sin_theta / 3, cos_theta + (1 - cos_theta) / 3]\n])\nresult = np.dot(imga, rotation_matrix.T)",
            "multiline": True
        })    
        return inputs

class Panner(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", {
                    "default": "x_shift = pint1\ny_shift = pint2\nheight, width = imga.shape[1:3]\nnew_x = np.clip(np.arange(width) + x_shift, 0, width - 1)\nnew_y = np.clip(np.arange(height) + y_shift, 0, height - 1)\nX, Y = np.meshgrid(new_x, new_y)\nresult = imga[:, Y, X, :]",
                    "multiline": True
                }),
            },
            "optional": {
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}),
                "pint1": ("INT", {"default": 0, "min": -8888888, "tooltip": "Optional(INT),X-Axis Pixel(s)."}),
                "pint2": ("INT", {"default": 0, "min": -8888888, "tooltip": "Optional(INT),Y-Axis Pixel(s)."}),
            },
        }
        # 隐藏 imgb, imgc, pfloat1, pfloat2, pfloat3
        for param in ['imgb', 'imgc', 'pfloat1', 'pfloat2', 'pfloat3']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default":"x_shift = -pint1\ny_shift = -pint2\nheight, width = imga.shape[1:3]\nnew_x = np.clip(np.arange(width) + x_shift, 0, width - 1)\nnew_y = np.clip(np.arange(height) + y_shift, 0, height - 1)\nX, Y = np.meshgrid(new_x, new_y)\nresult = imga[:, Y, X, :]", "multiline": True})
        return inputs

class Outline(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "pint1": ("INT", {"default": 1, "min": -32, "max": 32, "tooltip": "Outline Width ,Number of offset pixels"}),
                "pfloat1": ("FLOAT", {"default": 10.0, "min": 0.05, "max": 100, "step": 0.01, "tooltip": "Optional,Depth Map Subdivision."}), 

            },
            "optional": {
                "imga": ("IMAGE", {"tooltip": "可选，参考尺寸，默认 =(1,512,512,3)"})
            }
        }
        # 隐藏其他参数
        for param in ['imgb', 'imgc', 'pfloat2', 'pfloat3', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default":"imga = np.clip(imga * pfloat1,0,1)\nint_shift = abs(pint1)\nheight, width = imga.shape[1:3]\n# 左偏移\nimg_left = np.roll(imga, int_shift, axis=2)\nimg_left[:, :, :int_shift, :] = imga[:, :, :int_shift, :]\n# 右偏移\nimg_right = np.roll(imga, -int_shift, axis=2)\nimg_right[:, :, -int_shift:, :] = imga[:, :, -int_shift:, :]\n# 上偏移\nimg_up = np.roll(imga, int_shift, axis=1)\nimg_up[:, :int_shift, :, :] = imga[:, :int_shift, :, :]\n# 下偏移\nimg_down = np.roll(imga, -int_shift, axis=1)\nimg_down[:, -int_shift:, :, :] = imga[:, -int_shift:, :, :]\n# 求和并计算轮廓\nsumimgs = img_left + img_right + img_up + img_down\nredir = 0\nif pint1 < 0: redir = 1\nresult = (sumimgs / 4 - imga) * ((-1) ** redir)\nresult = np.ceil(result - 0.005) ", "multiline": True})
        return inputs

class Contant3Vector(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "pfloat1": ("FLOAT", {"default": 255.0, "min":0, "max": 255, "step": 1, "tooltip": "Optional,RGB::R(0~255)."}),
                "pfloat2": ("FLOAT", {"default": 0.0, "min": 0, "max": 255, "step": 1, "tooltip": "Optional,RGB::G(0~255)."}),
                "pfloat3": ("FLOAT", {"default": 0.0, "min": 0, "max": 255, "step": 1, "tooltip": "Optional,RGB::B(0~255)."}),
            },
            "optional": {
                "imga": ("IMAGE", {"tooltip": "可选，参考尺寸，默认 =(1,512,512,3)"})
            }
        }
        # 隐藏其他参数
        for param in ['imgb', 'imgc', 'pint1', 'pint2']:

            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default":"#height, width = 512,512\nheight, width = imga.shape[1:3]\nimg = np.ones((1, height, width, 3), dtype=np.float32)\n[R,G,B] =[pfloat1,pfloat2,pfloat3]\nresult = img *[R,G,B] /255 ", "multiline": True})
        return inputs

class Chroma_Key_Alpha(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", {
                    "default": "",
                    "multiline": True
                }),
                "pfloat1": ("FLOAT", {"default": 0.1, "min":0, "max": 1, "step": 0.01, "tooltip": "Choma threshold: pfloat1."}),
                "pfloat2": ("FLOAT", {"default": 20, "min": 0, "max": 100, "step": 0.01, "tooltip": "Soft Blend: pfloat2."}),
            },
            "optional": {
                "imga": ("IMAGE", {"tooltip": "image color"}),
                "imgb": ("IMAGE", {"tooltip": "chroma color"}),
            }
        }
        # 隐藏其他参数
        for param in [ 'imgc', 'pint1', 'pint2']:

            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {"default":"#Choma threshold: pfloat1\n#Soft Blend: pfloat2\nimgb =imgc + imgb[0, 1, 1, :]\nchomachannel = np.sum((imga * imgb), axis=-1)\nsoftmask = chomachannel - np.sum(imga , axis=-1)/3\nresult = 1- np.ceil(softmask - pfloat1) - softmask * pfloat2", "multiline": True})
        return inputs

class Desaturation(CustomScript):
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "formula": ("STRING", { 
                    "default": "result = imga", 
                    "multiline": True 
                }), 
            },
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "pfloat1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Optional,Floating-point parameter passed to the script."}), 
            },
        }
        # 隐藏 imgb, imgc, pfloat2, pfloat3, pint1, pint2
        for param in ['imgb', 'imgc', 'pfloat2', 'pfloat3', 'pint1', 'pint2']:
            if param in inputs.get('required', {}):
                del inputs['required'][param]
            if param in inputs.get('optional', {}):
                inputs['optional'][param] = (inputs['optional'][param][0], {**inputs['optional'][param][1], 'hidden': True})
        inputs["required"]["formula"] = ("STRING", {
            "default": "desaturate = np.sum(imga , axis=-1)/3\nmakergb = np.stack([desaturate, desaturate, desaturate], axis=-1)\nresult = makergb * pfloat1 + (1-pfloat1)*imga\n",
            "multiline": True
        })    
        return inputs
