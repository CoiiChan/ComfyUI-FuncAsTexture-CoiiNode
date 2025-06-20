try:
    from .uv_mapping import NODE_CLASS_MAPPINGS as UV_MAPPING_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as UV_MAPPING_NAMES
    from .rotator import NODE_CLASS_MAPPINGS as ROTATOR_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as ROTATOR_NAMES
    from .custom_script_numpy import NODE_CLASS_MAPPINGS as NUMPY_CLASSES, NODE_DISPLAY_NAME_MAPPINGS as NUMPY_NAMES
    from .basenodes.basemath import Add, Subtraction, Multiply, Divided, Max, Min, Dot, Distance,Power,HueShift,Panner,Outline,Lerp,Clamp,Ceil,Oneminus,Sine,DDX,Contant3Vector,ifFunction,Chroma_Key_Alpha,Desaturation
except ImportError as e:
    print(f"导入模块时出错: {e}")
    UV_MAPPING_CLASSES = {}
    UV_MAPPING_NAMES = {}
    ROTATOR_CLASSES = {}
    ROTATOR_NAMES = {}
    NUMPY_CLASSES = {}
    NUMPY_NAMES = {}
    Add = None
    Subtraction = None
    Multiply = None
    Divided = None
    Max = None
    Min = None
    Dot = None
    Distance = None
    Power = None
    HueShift = None
    Panner = None
    Outline = None
    Lerp = None
    Clamp = None
    Ceil = None
    Oneminus = None
    Sine = None
    DDX = None
    Contant3Vector = None
    ifFunction = None
    Chroma_Key_Alpha = None
    Desaturation = None


NODE_CLASS_MAPPINGS = {
    **UV_MAPPING_CLASSES,
    **ROTATOR_CLASSES,
    **NUMPY_CLASSES,
    "Add": Add,
    "Subtraction": Subtraction,
    "Multiply": Multiply,
    "Divided": Divided,
    "Max": Max,
    "Min": Min,
    "Dot": Dot,
    "Distance": Distance,
    "Power": Power,
    "HueShift": HueShift,
    "Panner": Panner,
    "Outline": Outline,
    "Lerp": Lerp,
    "Clamp": Clamp,
    "Ceil": Ceil,
    "Oneminus": Oneminus,
    "Sine": Sine,
    "DDX": DDX,
    "Contant3Vector": Contant3Vector,
    "ifFunction": ifFunction,
    "Chroma_Key_Alpha": Chroma_Key_Alpha,
    "Desaturation": Desaturation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **UV_MAPPING_NAMES,
    **ROTATOR_NAMES,
    **NUMPY_NAMES,
    "Add": "Add",
    "Subtraction": "Subtraction",
    "Multiply": "Multiply",
    "Divided": "Divided",  
    "Max": "Max",
    "Min": "Min",
    "Dot": "Dot",
    "Distance": "Distance",
    "Power": "Power",
    "HueShift": "HueShift",
    "Panner": "Panner",
    "Outline": "Outline",
    "Lerp": "Lerp",
    "Clamp": "Clamp",
    "Ceil": "Ceil",
    "Oneminus": "1-x",
    "Sine": "Sine",
    "DDX": "DDX",
    "Contant3Vector": "Contant3Vector(Color)",
    "ifFunction": "if (FuncAsTexture)",
    "Chroma_Key_Alpha": "Chroma_Key_Alpha",
    "Desaturation": "Desaturation",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
