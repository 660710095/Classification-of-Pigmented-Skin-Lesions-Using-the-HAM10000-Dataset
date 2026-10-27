"""
src/dataset.py
คลาส PyTorch Dataset และฟังก์ชันคำนวณค่าน้ำหนักคลาส (Class Weights / Sampler) สำหรับ HAM10000
"""

# นำเข้าโมดูล os สำหรับจัดการ path และระบบไฟล์พื้นฐาน
import os
# นำเข้า Path จาก pathlib สำหรับจัดการพาธไฟล์แบบทันสมัย
from pathlib import Path
# นำเข้า type hints สำหรับระบุประเภทของตัวแปรและฟังก์ชัน
from typing import Callable, Dict, List, Optional, Tuple, Union
# นำเข้า numpy สำหรับแปลง array ข้อมูลรูปภาพ
import numpy as np
# นำเข้า pandas สำหรับอ่านไฟล์ CSV ตาราง splits
import pandas as pd
# นำเข้า Image จาก PIL สำหรับเปิดและประมวลผลไฟล์ภาพ JPG
from PIL import Image
# นำเข้า torch สำหรับสร้างและจัดการ PyTorch Tensors
import torch
# นำเข้า Dataset และ WeightedRandomSampler จาก torch.utils.data
from torch.utils.data import Dataset, WeightedRandomSampler
# นำเข้า transforms จาก torchvision เพื่อแปลง Tensor กลับเป็น PIL Image
from torchvision import transforms

# ลิสต์ลำดับชื่อคลาสโรคผิวหนังทั้ง 7 คลาส เรียงตามลำดับตัวอักษรที่แน่นอน (Fixed Order)
CLASSES: List[str] = [
    "akiec",  # 0: Actinic keratoses and intraepithelial carcinoma
    "bcc",    # 1: Basal cell carcinoma
    "bkl",    # 2: Benign keratosis-like lesions
    "df",     # 3: Dermatofibroma
    "mel",    # 4: Melanoma
    "nv",     # 5: Melanocytic nevi
    "vasc",   # 6: Vascular lesions
]

# ดิกชันนารีแมปชื่อโรค -> ตัวเลขจำนวนเต็ม (0 ถึง 6) สำหรับเป็น Target Label ของโมเดล
CLASS_TO_IDX: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
# ดิกชันนารีแมปตัวเลขจำนวนเต็ม -> ชื่อโรค สำหรับแปลงผลลัพธ์การทำนายกลับเป็นชื่อโรคที่มนุษย์เข้าใจได้
IDX_TO_CLASS: Dict[int, str] = {idx: cls_name for idx, cls_name in enumerate(CLASSES)}


class HAM10000Dataset(Dataset):
    """
    คลาส Dataset สำหรับโหลดข้อมูล HAM10000 ใน PyTorch
    - อ่านข้อมูลจาก splits/splits.csv
    - ค้นหาตำแหน่งไฟล์ภาพอัตโนมัติจาก HAM10000_images_part_1 หรือ part_2
    - แปลงชื่อคลาส dx เป็นตัวเลขตามลำดับ CLASSES ที่กำหนดไว้
    - รองรับการ Cache รูปภาพลง RAM เป็น uint8 Tensor เพื่อประหยัดเวลาอ่านดิสก์ซ้ำๆ ในแต่ละ Epoch
    """

    def __init__(
        self,
        splits_csv: Union[str, Path] = "splits/splits.csv",  # พาธของไฟล์ splits.csv
        split: str = "train",                                # เลือกว่าจะโหลด split ไหน ('train', 'val', หรือ 'test')
        data_dir: Union[str, Path] = "data",                 # โฟลเดอร์หลักที่เก็บ part_1 และ part_2
        transform: Optional[Callable] = None,                # Pipeline การแปลงรูปภาพ (Data Augmentation)
        cache_in_ram: bool = False,                          # เปิดใช้งานการแคชภาพลง RAM หรือไม่
        img_size: int = 224,                                 # ขนาดภาพที่ใช้ตอน Resize ก่อนแคชลง RAM
    ):
        # เรียกคอนสตรักเตอร์ของคลาสแม่ (torch.utils.data.Dataset)
        super().__init__()
        
        # บันทึกพาธ splits.csv ลงในตัวแปรประจำคลาส
        self.splits_csv = Path(splits_csv)
        # แปลงชื่อ split ให้เป็นตัวพิมพ์เล็กและตัดช่องว่างส่วนเกิน
        self.split = split.lower().strip()
        # บันทึกโฟลเดอร์หลัก data_dir
        self.data_dir = Path(data_dir)
        # บันทึก transform pipeline ที่ส่งเข้ามา
        self.transform = transform
        # บันทึกสถานะการเปิด/ปิด cache_in_ram
        self.cache_in_ram = cache_in_ram
        # บันทึกขนาดภาพเป้าหมาย
        self.img_size = img_size

        # ตรวจสอบว่าไฟล์ splits.csv มีอยู่จริงหรือไม่ ถ้าไม่มีให้แจ้งเตือนให้รัน make_splits.py ก่อน
        if not self.splits_csv.exists():
            raise FileNotFoundError(
                f"ไม่พบไฟล์ Splits ที่: {self.splits_csv}. "
                "กรุณารันคำสั่ง `python scripts/make_splits.py` ก่อนเริ่มใช้งาน"
            )

        # อ่านตาราง splits.csv ด้วย pandas
        df_all = pd.read_csv(self.splits_csv)
        # ตรวจสอบว่าในไฟล์มีคอลัมน์ชื่อ 'split' หรือไม่
        if "split" not in df_all.columns:
            raise ValueError(f"ไฟล์ Splits {self.splits_csv} ไม่มีคอลัมน์ 'split'")

        # กำหนดชุดชื่อ split ที่ถูกต้อง
        valid_splits = {"train", "val", "test"}
        # ตรวจสอบว่าชื่อ split ที่ระบุเข้ามาถูกต้องหรือไม่
        if self.split not in valid_splits:
            raise ValueError(f"ชื่อ split '{self.split}' ไม่ถูกต้อง ต้องเป็นหนึ่งใน {valid_splits}")

        # กรองข้อมูลเฉพาะแถวที่อยู่ใน split ที่ต้องการ และรีเซ็ตหมายเลข index ของตาราง
        self.df = df_all[df_all["split"] == self.split].reset_index(drop=True)
        # หากไม่มีข้อมูลใน split ที่เลือก ให้แจ้งเตือน error
        if len(self.df) == 0:
            raise ValueError(f"ไม่พบข้อมูลสำหรับ split '{self.split}' ใน {self.splits_csv}")

        # รายชื่อโฟลเดอร์รูปภาพที่เป็นไปได้สำหรับ Part 1 (รองรับทั้งตัวพิมพ์เล็กและตัวพิมพ์ใหญ่)
        part1_candidates = [
            self.data_dir / "HAM10000_images_part_1",
            self.data_dir / "ham10000_images_part_1",
        ]
        # รายชื่อโฟลเดอร์รูปภาพที่เป็นไปได้สำหรับ Part 2
        part2_candidates = [
            self.data_dir / "HAM10000_images_part_2",
            self.data_dir / "ham10000_images_part_2",
        ]

        # ค้นหาว่าโฟลเดอร์ใดมีอยู่จริงในเครื่อง แล้วเก็บพาธจริงไว้ในลิสต์ self.image_dirs
        self.image_dirs: List[Path] = []
        for cand_list in [part1_candidates, part2_candidates]:
            for cand in cand_list:
                if cand.exists() and cand.is_dir():
                    self.image_dirs.append(cand)
                    break

        # หากไม่พบโฟลเดอร์ภาพเลย ให้หยุดและแจ้ง error ทันที
        if not self.image_dirs:
            raise FileNotFoundError(
                f"ไม่พบโฟลเดอร์ภาพใน '{self.data_dir}'. "
                "ต้องการ HAM10000_images_part_1 และ HAM10000_images_part_2"
            )

        # สร้างลิสต์สำหรับเก็บพาธของแต่ละรูปภาพใน split นี้
        self.image_paths: List[Path] = []
        # สร้างลิสต์สำหรับเก็บตัวเลข label (0-6) ของแต่ละรูปภาพ
        self.labels: List[int] = []

        # วนลูปอ่านข้อมูลทีละแถวเพื่อตรวจสอบไฟล์ภาพและจัดเตรียมข้อมูล
        for _, row in self.df.iterrows():
            img_id = str(row["image_id"]).strip()  # ชื่อรหัสภาพ เช่น ISIC_0024306
            dx_class = str(row["dx"]).strip()      # ชื่อโรค เช่น mel, nv

            # ตรวจสอบว่าชื่อโรคอยู่ใน 7 คลาสหรือไม่
            if dx_class not in CLASS_TO_IDX:
                raise ValueError(f"พบคลาสที่ไม่รู้จัก '{dx_class}' ของรูป '{img_id}'")
            # แปลงชื่อโรคเป็นหมายเลข index (0 ถึง 6) แล้วเพิ่มลงในลิสต์ self.labels
            self.labels.append(CLASS_TO_IDX[dx_class])

            # เติม .jpg หากไม่มี เพื่อใช้ค้นหาไฟล์ในดิสก์
            filename = f"{img_id}.jpg" if not img_id.endswith(".jpg") else img_id
            resolved_path = None
            
            # ตรวจสอบว่าไฟล์ภาพนี้อยู่ใน part 1 หรือ part 2
            for img_dir in self.image_dirs:
                cand_path = img_dir / filename
                if cand_path.is_file():
                    resolved_path = cand_path
                    break

            # หากค้นหาทั้งสองโฟลเดอร์แล้วไม่พบไฟล์ภาพ ให้แจ้งเตือน error ทันที
            if resolved_path is None:
                raise FileNotFoundError(
                    f"ไม่พบไฟล์รูป '{filename}' สำหรับรหัส '{img_id}' ใน {self.image_dirs}"
                )
            # บันทึกพาธของรูปภาพที่พบลงในลิสต์ self.image_paths
            self.image_paths.append(resolved_path)

        # ตัวแปรเก็บเทนเซอร์ที่แคชไว้ใน RAM (ค่าเริ่มต้นเป็น None)
        self.cached_tensors: Optional[List[torch.Tensor]] = None
        # หากเปิดตัวเลือก cache_in_ram ให้ทำการพรีโหลดและแคชรูปภาพทันที
        if self.cache_in_ram:
            self._cache_images_in_ram()

    def _cache_images_in_ram(self) -> None:
        """
        โหลดและ Resize ภาพล่วงหน้าให้มีขนาด (img_size, img_size)
        แล้วแปลงเป็น PyTorch Tensor ชนิด uint8 ขนาด (3, H, W) เก็บไว้ใน RAM
        ซึ่งใช้แรมเพียง ~150 KB ต่อรูป (ทั้ง split ใช้แรมเพียง ~1.05 GB)
        """
        print(f"[{self.split.upper()}] กำลังแคชรูปภาพ {len(self.image_paths)} รูปลง RAM ที่ขนาด {self.img_size}x{self.img_size} (uint8)...")
        cached = []
        for path in self.image_paths:
            # เปิดรูปภาพจากดิสก์ด้วย PIL
            with Image.open(path) as img:
                # แปลงโหมดสีเป็น RGB ป้องกันภาพขาวดำหรือ RGBA ปนเปื้อน
                img_rgb = img.convert("RGB")
                # ปรับขนาดภาพล่วงหน้าด้วย Bilinear interpolation
                img_resized = img_rgb.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
                # แปลงภาพ PIL เป็น numpy array ชนิด uint8 (H, W, 3) ค่า 0-255
                arr = np.array(img_resized, dtype=np.uint8)
                # แปลงเป็น PyTorch Tensor และสลับมิติเป็น (C, H, W) คือ (3, H, W)
                tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
                # เก็บเทนเซอร์ลงในลิสต์แคช
                cached.append(tensor)
        
        # บันทึกลิสต์เทนเซอร์ที่แคชเสร็จสมบูรณ์ลงในตัวแปรคลาส
        self.cached_tensors = cached
        # คำนวณปริมาณการใช้งานแรมเป็นเมกะไบต์ (MB)
        mem_mb = (len(cached) * 3 * self.img_size * self.img_size) / (1024 * 1024)
        print(f"[{self.split.upper()}] แคชเสร็จสิ้น {len(cached)} รูปบน RAM (ใช้แรมประมาณ ~{mem_mb:.1f} MB)")

    def __len__(self) -> int:
        """ส่งกลับจำนวนภาพทั้งหมดที่มีใน Split นี้"""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        ดึงข้อมูลลำดับที่ idx ออกมาในรูปแบบ (image_tensor, label)
        - หากใช้ cache_in_ram: ดึง uint8 tensor จากแรมมาทำ to_pil_image แล้วส่งต่อให้ transform
        - หากไม่ใช้แคช: เปิดไฟล์ภาพสดจากดิสก์แล้วส่งให้ transform
        """
        if self.cache_in_ram and self.cached_tensors is not None:
            # ดึง uint8 tensor จากแคชในแรม (ขนาด 3, H, W)
            uint8_tensor = self.cached_tensors[idx]
            # แปลง uint8 tensor กลับเป็น PIL Image ชั่วคราวในแรมแบบเร็ว เพื่อให้รองรับ torchvision transforms
            img = transforms.functional.to_pil_image(uint8_tensor)
        else:
            # เปิดไฟล์รูปภาพจากฮาร์ดดิสก์และแปลงเป็น RGB
            img = Image.open(self.image_paths[idx]).convert("RGB")

        # นำรูปภาพผ่าน Augmentation / Normalization Pipeline ตามที่กำหนดไว้
        if self.transform is not None:
            img = self.transform(img)

        # ดึงค่า label ตัวเลขของรูปภาพลำดับที่ idx
        label = self.labels[idx]
        # ส่งกลับเป็นคู่ Tuple (เทนเซอร์ภาพ, ตัวเลขคลาส)
        return img, label


def get_class_weights(
    splits_csv: Union[str, Path] = "splits/splits.csv",
    split: str = "train"
) -> torch.Tensor:
    """
    คำนวณค่าน้ำหนักประจำแต่ละคลาสแบบ Balanced Class Weights จากชุด Train
    ตามสูตรมาตรฐาน: weight_c = N_total / (N_classes * N_c)
    ใช้สำหรับส่งให้ฟังก์ชัน CrossEntropyLoss(weight=weights) เพื่อแก้ปัญหา Class Imbalance
    """
    # แปลง path เป็น Path object
    path = Path(splits_csv)
    # ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ Splits ที่: {splits_csv}")

    # อ่านไฟล์ splits.csv
    df = pd.read_csv(path)
    # กรองเฉพาะแถวที่อยู่ใน split ที่ต้องการ (ปกติคือชุด train)
    split_df = df[df["split"] == split]
    # หากไม่มีข้อมูล ให้แจ้งเตือน error
    if len(split_df) == 0:
        raise ValueError(f"ไม่พบข้อมูลสำหรับ split '{split}' ใน {splits_csv}")

    # นับจำนวนภาพของแต่ละคลาสโรค
    counts = split_df["dx"].value_counts()
    # จำนวนภาพทั้งหมดในชุดนี้
    n_samples = len(split_df)
    # จำนวนคลาสทั้งหมด (7 คลาส)
    n_classes = len(CLASSES)

    # วนลูปคำนวณค่าน้ำหนักตามลำดับคลาสใน CLASSES
    weights = []
    for cls_name in CLASSES:
        count = counts.get(cls_name, 0)
        # ป้องกันการหารด้วยศูนย์หากคลาสนั้นไม่มีรูปเลย
        if count == 0:
            raise ValueError(f"คลาส '{cls_name}' ไม่มีข้อมูลภาพใน split '{split}'")
        # คำนวณน้ำหนักผกผันตามความถี่: คลาสที่มีรูปน้อยจะได้ค่าน้ำหนักสูง
        w = n_samples / (n_classes * count)
        weights.append(w)

    # ส่งกลับผลลัพธ์เป็น 1D FloatTensor ขนาด (7,)
    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(
    dataset_or_splits_csv: Union[Dataset, str, Path],
    split: str = "train",
) -> WeightedRandomSampler:
    """
    สร้าง WeightedRandomSampler สำหรับสุ่มตัวอย่างข้อมูลเข้า Batch ให้แต่ละคลาสมีโอกาสถูกเลือกเท่าๆ กัน
    ใช้ส่งเข้า DataLoader(dataset, sampler=sampler) แทน shuffle=True
    """
    # ตรวจสอบว่าอินพุตเป็นออบเจกต์ HAM10000Dataset หรือเป็นพาธไฟล์ CSV
    if isinstance(dataset_or_splits_csv, HAM10000Dataset):
        labels = dataset_or_splits_csv.labels
    else:
        path = Path(dataset_or_splits_csv)
        df = pd.read_csv(path)
        split_df = df[df["split"] == split]
        labels = [CLASS_TO_IDX[dx] for dx in split_df["dx"]]

    # นับความถี่ของแต่ละ label
    label_series = pd.Series(labels)
    class_counts = label_series.value_counts().to_dict()

    # กำหนดค่าน้ำหนักของแต่ละรูปภาพให้ผกผันกับจำนวนรูปในคลาสนั้น (1.0 / count)
    sample_weights = [1.0 / class_counts[lbl] for lbl in labels]
    # แปลงค่าน้ำหนักตัวอย่างเป็น PyTorch DoubleTensor
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)

    # สร้างและส่งกลับ WeightedRandomSampler ที่อนุญาตให้สุ่มซ้ำได้ (replacement=True)
    return WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True,
    )
