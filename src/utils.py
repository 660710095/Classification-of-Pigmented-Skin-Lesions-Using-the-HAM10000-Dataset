"""
src/utils.py
ฟังก์ชันอรรถประโยชน์สำหรับตรวจสอบสภาพแวดล้อมฮาร์ดแวร์ GPU/CUDA,
การตั้งค่า Random Seed เพื่อให้ผลการทดลองซ้ำได้ (Reproducibility),
การโหลดไฟล์การตั้งค่า YAML, และการเปิดใช้งาน Mixed Precision (bfloat16)
"""

# นำเข้า os สำหรับจัดการ Environment Variables ของระบบปฏิบัติการ
import os
# นำเข้า random สำหรับสุ่มตัวเลขในไลบรารีมาตรฐานของ Python
import random
# นำเข้า Path จาก pathlib สำหรับจัดการ file path
from pathlib import Path
# นำเข้า type hints สำหรับระบุชนิดข้อมูล
from typing import Any, Dict, Optional
# นำเข้า numpy สำหรับตั้งค่า seed ของ NumPy
import numpy as np
# นำเข้า torch สำหรับจัดการ PyTorch และ GPU CUDA
import torch
# นำเข้า yaml (PyYAML) สำหรับอ่านไฟล์การตั้งค่า config.yaml
import yaml


def set_seed(seed: int = 42) -> None:
    """
    ตั้งค่า Random Seed ให้กับ Python, NumPy, และ PyTorch ทุกส่วน
    เพื่อให้ผลการรันการทดลองและการแบ่งข้อมูลออกมาเหมือนเดิมทุกครั้ง
    """
    # กำหนด seed ให้กับฟังก์ชัน random มาตรฐานของ Python
    random.seed(seed)
    # กำหนด seed ให้กับ NumPy
    np.random.seed(seed)
    # กำหนด seed ให้กับ Hash function ในระบบ Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    # กำหนด seed ให้กับ PyTorch บน CPU
    torch.manual_seed(seed)
    
    # หากมี GPU (CUDA) ให้กำหนด seed ให้กับ GPU ทุกตัวในเครื่อง
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # บังคับให้อัลกอริทึมของ cuDNN ทำงานแบบเจาะจงผลลัพธ์แน่นอน (deterministic)
        torch.backends.cudnn.deterministic = True
        # ปิด benchmark เพื่อไม่ให้ cuDNN เปลี่ยนอัลกอริทึมระหว่างรัน
        torch.backends.cudnn.benchmark = False


def print_env_info() -> None:
    """
    พิมพ์ข้อมูลสรุปเวอร์ชันของ PyTorch, CUDA, สเปกการ์ดจอ และ VRAM ทั้งหมด
    มีประโยชน์มากสำหรับเซิร์ฟเวอร์ GPU ที่ใช้งานร่วมกันหลายคนในทีม
    """
    print("=" * 60)
    print("ENVIRONMENT & HARDWARE SUMMARY")
    print("=" * 60)
    # แสดงเวอร์ชันของ PyTorch ที่ติดตั้งในเครื่อง
    print(f"PyTorch Version : {torch.__version__}")
    
    # ตรวจสอบว่าสามารถเข้าถึง GPU CUDA ได้หรือไม่
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available  : {cuda_available}")

    if cuda_available:
        # แสดงเวอร์ชัน CUDA ที่ PyTorch คอมไพล์มาด้วย
        print(f"CUDA Version    : {torch.version.cuda}")
        # แสดงจำนวนการ์ดจอที่มองเห็นในระบบ
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count       : {gpu_count}")
        # ดึงหมายเลข index ของการ์ดจอปัจจุบันที่กำลังใช้งาน
        current_device = torch.cuda.current_device()
        # ดึงชื่อรุ่นการ์ดจอ (เช่น NVIDIA RTX PRO 6000)
        device_name = torch.cuda.get_device_name(current_device)
        # คำนวณขนาดหน่วยความจำ VRAM ทั้งหมดในหน่วยกิกะไบต์ (GB)
        total_vram_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)
        print(f"Current Device  : cuda:{current_device} ({device_name})")
        print(f"Total VRAM      : {total_vram_gb:.2f} GB")
        # ตรวจสอบว่าฮาร์ดแวร์การ์ดจอนี้รองรับ bfloat16 หรือไม่
        bf16_supported = torch.cuda.is_bf16_supported()
        print(f"bfloat16 Support: {bf16_supported}")
    else:
        # กรณีรันบน CPU เท่านั้น
        print("Device          : CPU only")
    print("=" * 60)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    อ่านและแปลงไฟล์คอนฟิก YAML ให้เป็น Python Dictionary
    """
    path = Path(config_path)
    # ตรวจสอบว่าไฟล์คอนฟิกมีอยู่จริงหรือไม่
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ Config ที่: {config_path}")

    # เปิดไฟล์และอ่านข้อมูลด้วย PyYAML safe_load
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_amp_context(use_amp: bool = True, dtype: torch.dtype = torch.bfloat16):
    """
    ส่งกลับ Context Manager สำหรับ Automatic Mixed Precision (torch.autocast)
    เพื่อช่วยประหยัดแรม VRAM บนการ์ดจอ RTX 6000 และช่วยให้เทรนเร็วขึ้นเกือบเท่าตัว
    """
    # กำหนดประเภทอุปกรณ์ว่าเป็น cuda หรือ cpu
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    
    # หากการ์ดจอไม่รองรับ bfloat16 ให้สลับไปใช้ float16 อัตโนมัติเพื่อความปลอดภัย
    if device_type == "cuda" and not torch.cuda.is_bf16_supported() and dtype == torch.bfloat16:
        dtype = torch.float16
        
    # ส่งกลับ autocast context ตามประเภทอุปกรณ์และ dtype ที่ระบุ
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=use_amp)
