"""
src/models/__init__.py
ระบบ Model Registry แบบกระจายศูนย์สำหรับทีม 4 คน
เพื่อนร่วมทีมสามารถเพิ่มไฟล์โมเดลของตนเองใน src/models/<model_name>.py
และลงทะเบียนด้วย @register_model ได้ทันทีโดยไม่ต้องแก้ไขไฟล์นี้ร่วมกัน (ลดปัญหา Git Conflict)
"""

# นำเข้า importlib สำหรับนำเข้า (import) โมดูลไฟล์ Python แบบไดนามิกในขณะทำงาน
import importlib
# นำเข้า pkgutil สำหรับสำรวจโมดูลและแพ็กเกจย่อยในโฟลเดอร์
import pkgutil
# นำเข้า Path จาก pathlib สำหรับการค้นหาไฟล์ .py ภายในไดเรกทอรี
from pathlib import Path
# นำเข้า type hints สำหรับระบุประเภทของตัวแปรและฟังก์ชัน
from typing import Callable, Dict, List, Optional
# นำเข้า torch.nn สำหรับกำหนดชนิดผลลัพธ์ของโมเดล Neural Network (nn.Module)
import torch.nn as nn

# ดิกชันนารีกลางสำหรับจัดเก็บความสัมพันธ์: ชื่อโมเดล (str) -> ฟังก์ชันสร้างโมเดล (Builder Function)
_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}

# ตัวแปรสถานะ (Flag) เพื่อระบุว่าได้ทำการค้นหาและนำเข้าโมดูลทั้งหมดในโฟลเดอร์แล้วหรือยัง
_DISCOVERED: bool = False


def register_model(name: Optional[str] = None):
    """
    Decorator สำหรับลงทะเบียนฟังก์ชันสร้างโมเดลเข้าสู่ Registry กลาง

    การใช้งาน:
        @register_model("resnet50")
        def build_resnet50(num_classes=7, pretrained=True, **kwargs):
            ...
    """
    def decorator(fn: Callable[..., nn.Module]):
        # หากไม่ได้ระบุชื่อ name ให้ใช้ชื่อฟังก์ชันนั้นเป็นชื่อโมเดลอัตโนมัติ
        model_name = name if name is not None else fn.__name__
        
        # ตรวจสอบว่าชื่อโมเดลนี้เคยถูกลงทะเบียนไปแล้วหรือไม่ เพื่อป้องกันการตั้งชื่อซ้ำ
        if model_name in _MODEL_REGISTRY:
            raise ValueError(f"โมเดลชื่อ '{model_name}' ถูกลงทะเบียนในระบบไปแล้ว")
        
        # บันทึกฟังก์ชันสร้างโมเดลลงใน _MODEL_REGISTRY โดยใช้ model_name เป็น key
        _MODEL_REGISTRY[model_name] = fn
        return fn

    # กรณีที่ผู้ใช้เรียก @register_model โดยไม่มีวงเล็บ parameter
    if callable(name):
        fn = name
        name = fn.__name__
        return decorator(fn)
    
    return decorator


def _discover_models() -> None:
    """
    ฟังก์ชันค้นหาและนำเข้า (auto-import) ไฟล์ .py ทั้งหมดที่อยู่ใน src/models/ แบบอัตโนมัติ
    เพื่อให้โค้ด @register_model ในแต่ละไฟล์ทำงานและลงทะเบียนตัวเองเข้าสู่ระบบ
    """
    global _DISCOVERED
    # ถ้าเคยทำการค้นหาไปแล้วในเซสชันนี้ ให้ข้ามไปทันทีเพื่อประสิทธิภาพ
    if _DISCOVERED:
        return

    # หาตำแหน่งโฟลเดอร์จริงของ src/models
    package_dir = Path(__file__).resolve().parent
    
    # วนลูปหาไฟล์ .py ทั้งหมดในไดเรกทอรีนี้
    for file_path in package_dir.glob("*.py"):
        module_name = file_path.stem  # ดึงเฉพาะชื่อไฟล์โดยไม่รวมนามสกุล .py
        
        # ข้ามไฟล์ที่ขึ้นต้นด้วย _ เช่น __init__.py หรือไฟล์ชั่วคราว
        if module_name.startswith("_"):
            continue
        
        # กำหนดชื่อโมดูลแบบเต็ม เช่น src.models.resnet50
        full_module_name = f"{__name__}.{module_name}"
        
        # หากยังไม่เคยถูก import มาก่อน ให้ทำการ import โมดูลนั้นเข้ามา
        if full_module_name not in importlib.sys.modules:
            try:
                importlib.import_module(full_module_name)
            except Exception as e:
                # หากไฟล์ของเพื่อนร่วมทีมคนใดมี syntax หรือ import error ให้แสดง Warning เพื่อแจ้งเตือน
                import warnings
                warnings.warn(f"ไม่สามารถนำเข้าไฟล์โมเดล '{full_module_name}': {e}")

    # อัปเดตสถานะว่าค้นหาและลงทะเบียนเรียบร้อยแล้ว
    _DISCOVERED = True


def get_model(
    name: str,
    num_classes: int = 7,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    ดึงและสร้างอินสแตนซ์ของโมเดลตามชื่อที่ลงทะเบียนไว้ใน Registry

    พารามิเตอร์:
        name: ชื่อโมเดลที่ต้องการสร้าง (เช่น 'resnet50', 'efficientnet_b0')
        num_classes: จำนวนคลาสเอาต์พุต (ค่าเริ่มต้น: 7 สำหรับ HAM10000)
        pretrained: โหลดค่าน้ำหนักที่ผ่านการฝึกฝนมาก่อนหรือไม่ (ค่าเริ่มต้น: True)
        **kwargs: พารามิเตอร์เพิ่มเติมที่จะส่งต่อไปยังฟังก์ชันสร้างโมเดล

    ส่งกลับ:
        โมเดล PyTorch (nn.Module) ที่พร้อมนำไปใช้งาน
    """
    # สแกนหาโมเดลใหม่ทั้งหมดในโฟลเดอร์ก่อน
    _discover_models()

    # ตรวจสอบว่ามีชื่อโมเดลที่เรียกหาอยู่ใน Registry หรือไม่
    if name not in _MODEL_REGISTRY:
        available = list_models()
        err_msg = (
            f"ไม่พบโมเดล '{name}' ในระบบ Registry! "
            f"โมเดลที่มีอยู่ตอนนี้: {available if available else 'ยังไม่มี (เพิ่มโมเดลได้ที่ src/models/<name>.py)'}"
        )
        raise KeyError(err_msg)

    # ดึงฟังก์ชันสร้างโมเดลจาก Registry
    builder = _MODEL_REGISTRY[name]
    # เรียกฟังก์ชันเพื่อสร้างและส่งกลับออบเจกต์โมเดล
    return builder(num_classes=num_classes, pretrained=pretrained, **kwargs)


def list_models() -> List[str]:
    """
    แสดงรายชื่อโมเดลทั้งหมดที่ถูกลงทะเบียนไว้ในระบบ (เรียงตามตัวอักษร)
    """
    # สแกนหาโมเดลทั้งหมดก่อน
    _discover_models()
    # ส่งกลับรายชื่อโมเดลที่ลงทะเบียนแล้ว
    return sorted(list(_MODEL_REGISTRY.keys()))


# กำหนดสัญลักษณ์ที่อนุญาตให้เรียกใช้เมื่อ import * จากแพ็กเกจนี้
__all__ = ["register_model", "get_model", "list_models"]
