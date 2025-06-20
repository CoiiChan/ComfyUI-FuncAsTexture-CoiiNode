from PIL import Image
import torch
import numpy as np

class CustomScript:
    """
    A custom node to apply mathematical operations on selected channels of two input images using a multi-line formula.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return { 
            "required": { 
                "formula": ("STRING", { 
                    "default": "#Default already import numpy as np\nimga_r = (imga_r + imga_g) * 0.5\nimga = make_rgb(imga_r, imga_g, imga_b)\nresult = imga\n#Node Introduction:https://github.com/CoiiChan/Comfyui-FuncAsTexture-CoiiNode", 
                    "multiline": True, 
                }), 
            }, 
            "optional": { 
                "imga": ("IMAGE", {"tooltip": "Optional ,Reference size ,Default =(1,512,512,3)"}), 
                "imgb": ("IMAGE", {"tooltip": "Optional"}), 
                "imgc": ("IMAGE", {"tooltip": "Optional"}), 
                "pfloat1": ("FLOAT", {"default": 0.0, "min": -8888888.0, "step": 0.01, "tooltip": "Optional,Floating-point parameter passed to the script."}), 
                "pfloat2": ("FLOAT", {"default": 0.0, "min": -8888888.0, "step": 0.01, "tooltip": "Optional,Floating-point parameter passed to the script."}), 
                "pfloat3": ("FLOAT", {"default": 0.0, "min": -8888888.0, "step": 0.01, "tooltip": "Optional,Floating-point parameter passed to the script."}), 
                "pint1": ("INT", {"default": 0, "min": -8888888, "tooltip": "Optional(INT),Integer parameter passed to the script."}), 
                "pint2": ("INT", {"default": 0, "min": -8888888, "tooltip": "Optional(INT),Integer parameter passed to the script."}), 
            }, 

        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "FunctionAsTexture"

    def execute(self, formula, imga=None, imgb=None, imgc=None, pfloat1=0.0, pfloat2=0.0, pfloat3=0.0, pint1=0, pint2=0):
        # 处理 imga 为 None 的情况
        if imga is None:
            img_a = np.zeros((1, 512, 512, 3), dtype=np.float32)
        else:
            img_a = imga.cpu().numpy()
        
        # Handle case when imgb is not provided
        if imgb is not None:
            img_b = imgb.cpu().numpy()
        else:
            # Create a zero array with same shape as img_a if imgb is None
            img_b = np.zeros_like(img_a)

        # Handle case when imgc is not provided
        if imgc is not None:
            img_c = imgc.cpu().numpy()
        else:
            # Create a zero array with same shape as img_a if imgc is None
            img_c = np.zeros_like(img_a)

        # 创建局部作用域并填充初始变量
        local_scope = { 
            "np": np, 
            "imga": img_a.copy(),  
            "imgb": img_b.copy(),
            "imgc": img_c.copy(),
            "imga_r": img_a[..., 0].copy(),
            "imga_g": img_a[..., 1].copy(),
            "imga_b": img_a[..., 2].copy(),
            "imgb_r": img_b[..., 0].copy(),
            "imgb_g": img_b[..., 1].copy(),
            "imgb_b": img_b[..., 2].copy(),
            "imgc_r": img_c[..., 0].copy(),
            "imgc_g": img_c[..., 1].copy(),
            "imgc_b": img_c[..., 2].copy(),
            "pfloat1": pfloat1,
            "pfloat2": pfloat2,
            "pfloat3": pfloat3,
            "pint1": pint1,
            "pint2": pint2,
            
        }

        # 定义辅助函数，用于重新组合通道
        def makergb(r, g, b):
            if r.shape != g.shape or r.shape != b.shape:
                raise ValueError("Channels must have the same shape.")
            return np.stack([r, g, b], axis=-1)

        # 将辅助函数添加到局部作用域
        local_scope["make_rgb"] = makergb

        # exec custom function from formula
        try:
            exec(formula, {}, local_scope)
        except Exception as e:
            raise ValueError(f"Error evaluating formula: {formula}. Details: {e}")


        result = local_scope.get("result")

        if result is None:
            result = img_a.copy()

        if len(result.shape) == 3: 
            # 确保输入是3D (H,W,C) 格式
            if result.shape[2] == 1:  
                result = np.repeat(result, 3, axis=-1)
            else:  
                result = np.repeat(np.expand_dims(result, -1), 3, axis=-1)
        
        return (torch.from_numpy(result).float(),)




NODE_CLASS_MAPPINGS = {
    "CustomScriptNumpy": CustomScript,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomScriptNumpy": "CustomScript-NumPy",
}