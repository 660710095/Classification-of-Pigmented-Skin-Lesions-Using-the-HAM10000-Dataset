#!/usr/bin/env python3
"""
scripts/make_splits.py
สร้าง Train/Val/Test Split (70/15/15) สำหรับ HAM10000 Dataset
โดยแบ่งแบบ Grouped by lesion_id และรักษาสัดส่วนคลาส (StratifiedGroupKFold)
"""

# นำเข้า argparse เพื่อรองรับการรับ argument จาก command line (เช่น --metadata, --output, --seed)
import argparse
# นำเข้า Path จาก pathlib สำหรับจัดการ file path แบบ object-oriented และข้าม platform ได้สะดวก
from pathlib import Path
# นำเข้า pandas สำหรับอ่านและประมวลผลไฟล์ CSV ตารางข้อมูล
import pandas as pd
# นำเข้า StratifiedGroupKFold จาก scikit-learn สำหรับแบ่งกลุ่มข้อมูลแบบคงสัดส่วนคลาสและแยกกลุ่ม lesion_id ไม่ให้ปนกัน
from sklearn.model_selection import StratifiedGroupKFold


def make_splits(
    metadata_path: str = "data/HAM10000_metadata.csv",  # ที่อยู่ไฟล์ metadata ต้นฉบับ
    output_path: str = "splits/splits.csv",             # ที่อยู่ไฟล์ผลลัพธ์ที่จะบันทึก split
    seed: int = 42,                                     # random seed เพื่อให้ผลลัพธ์เหมือนเดิมทุกครั้ง (reproducibility)
    n_splits: int = 20,                                 # จำนวน fold ทั้งหมด (20 folds เพื่อแบ่ง 14:3:3 = 70%:15%:15%)
) -> pd.DataFrame:
    """
    ฟังก์ชันแบ่งข้อมูล HAM10000 เป็น Train (70%), Val (15%), Test (15%)
    โดยใช้ StratifiedGroupKFold 20 folds
    """
    # แปลง metadata_path สตริงเป็น Path object เพื่อตรวจสอบและใช้งาน
    meta_path = Path(metadata_path)
    
    # ตรวจสอบว่าไฟล์ metadata มีอยู่จริงหรือไม่ ถ้าไม่พบให้ raise FileNotFoundError ทันที
    if not meta_path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ Metadata ที่: {metadata_path}")

    # อ่านไฟล์ metadata.csv เข้าสู่ pandas DataFrame
    df = pd.read_csv(meta_path)
    # นับจำนวนภาพทั้งหมดในตาราง metadata
    total_images = len(df)
    # แสดงข้อความแจ้งจำนวนแถวข้อมูลที่โหลดมาได้
    print(f"โหลดข้อมูล metadata จาก {metadata_path} สำเร็จ: ทั้งหมด {total_images} รูป")

    # กำหนดคอลัมน์สำคัญที่จำเป็นต้องมีในไฟล์ metadata
    required_cols = {"image_id", "lesion_id", "dx"}
    # ตรวจหาว่ามีคอลัมน์ใดขาดหายไปหรือไม่
    missing = required_cols - set(df.columns)
    # หากขาดคอลัมน์สำคัญ ให้แจ้ง error ทันที
    if missing:
        raise ValueError(f"ไฟล์ Metadata ขาดคอลัมน์ที่จำเป็น: {missing}")

    # สร้างออบเจกต์ StratifiedGroupKFold จำนวน 20 folds เปิด shuffle=True และกำหนด seed=42
    # เหตุผลที่ใช้ 20 folds:
    # - 14 folds จาก 20 = 70% สำหรับ Train
    # - 3 folds จาก 20  = 15% สำหรับ Val
    # - 3 folds จาก 20  = 15% สำหรับ Test
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # สร้าง dictionary เปล่าเพื่อเก็บการจับคู่ index ข้อมูล -> หมายเลข fold (0 ถึง 19)
    fold_map = {}
    # วนลูปสร้าง split โดยส่ง y=dx (คลาสของโรค) และ groups=lesion_id (กลุ่มรอยโรคเดียวกัน)
    for fold_idx, (_, val_indices) in enumerate(sgkf.split(df, y=df["dx"], groups=df["lesion_id"])):
        # บันทึก index ของแต่ละรูปภาพว่าจัดอยู่ใน fold ใด
        for idx in val_indices:
            fold_map[idx] = fold_idx

    # เพิ่มคอลัมน์ 'fold' ลงใน DataFrame โดยแมปจาก fold_map
    df["fold"] = df.index.map(fold_map)

    # ฟังก์ชันย่อยสำหรับแปลง fold index (0-19) เป็นชื่อ split ('test', 'val', 'train')
    def assign_split(f: int) -> str:
        if f < 3:
            return "test"   # fold 0, 1, 2 รวมกันได้ 3/20 = 15% เป็น Test Set
        elif f < 6:
            return "val"    # fold 3, 4, 5 รวมกันได้ 3/20 = 15% เป็น Validation Set
        else:
            return "train"  # fold 6 ถึง 19 รวมกันได้ 14/20 = 70% เป็น Training Set

    # สร้างคอลัมน์ 'split' จากฟังก์ชัน assign_split
    df["split"] = df["fold"].map(assign_split)

    # คัดเลือกเฉพาะ 4 คอลัมน์ที่กำหนดในโจทย์: image_id, lesion_id, dx, split
    splits_df = df[["image_id", "lesion_id", "dx", "split"]].copy()

    # =========================================================================
    # ตรวจสอบ Data Leakage และความถูกต้องของข้อมูล (Assertion Checks)
    # =========================================================================
    # ดึงชุด lesion_id ที่ไม่ซ้ำกันของแต่ละ split มาไว้ใน set
    train_lesions = set(splits_df[splits_df["split"] == "train"]["lesion_id"])
    val_lesions = set(splits_df[splits_df["split"] == "val"]["lesion_id"])
    test_lesions = set(splits_df[splits_df["split"] == "test"]["lesion_id"])

    # ตรวจสอบการทับซ้อน (Intersection) ระหว่างแต่ละ split
    leak_train_val = train_lesions.intersection(val_lesions)   # การทับซ้อนระหว่าง Train กับ Val
    leak_train_test = train_lesions.intersection(test_lesions) # การทับซ้อนระหว่าง Train กับ Test
    leak_val_test = val_lesions.intersection(test_lesions)     # การทับซ้อนระหว่าง Val กับ Test

    # พิมพ์ผลการตรวจสอบความปลอดภัยของข้อมูล
    print("\n" + "=" * 65)
    print("DATASET LEAKAGE & INTEGRITY VERIFICATION")
    print("=" * 65)
    print(f"Total images                : {len(splits_df)}")
    print(f"Unique lesions              : {splits_df['lesion_id'].nunique()}")
    print(f"Train lesions count         : {len(train_lesions)}")
    print(f"Val lesions count           : {len(val_lesions)}")
    print(f"Test lesions count          : {len(test_lesions)}")
    print(f"Train & Val lesion overlap  : {len(leak_train_val)}")
    print(f"Train & Test lesion overlap : {len(leak_train_test)}")
    print(f"Val & Test lesion overlap   : {len(leak_val_test)}")

    # ตรวจสอบอย่างเข้มงวดด้วยคำสั่ง assert หากผิดเงื่อนไขโปรแกรมจะหยุดทำงานทันที
    # 1. ตรวจสอบว่าจำนวนรูปทั้งหมดเท่ากับ 10,015 รูป
    assert len(splits_df) == 10015, f"จำนวนภาพไม่ครบ 10,015 รูป (พบ {len(splits_df)})"
    # 2. ตรวจสอบว่าไม่มี lesion_id ซ้ำซ้อนระหว่าง Train และ Val
    assert len(leak_train_val) == 0, f"พบรอยโรคซ้ำระหว่าง Train และ Val: {len(leak_train_val)}"
    # 3. ตรวจสอบว่าไม่มี lesion_id ซ้ำซ้อนระหว่าง Train และ Test
    assert len(leak_train_test) == 0, f"พบรอยโรคซ้ำระหว่าง Train และ Test: {len(leak_train_test)}"
    # 4. ตรวจสอบว่าไม่มี lesion_id ซ้ำซ้อนระหว่าง Val และ Test
    assert len(leak_val_test) == 0, f"พบรอยโรคซ้ำระหว่าง Val และ Test: {len(leak_val_test)}"

    print(">> SUCCESS: Zero lesion leakage across splits. Assertions passed!")
    print("=" * 65)

    # =========================================================================
    # แสดงจำนวนและสัดส่วนคลาสของแต่ละ Split
    # =========================================================================
    print("\n" + "=" * 65)
    print("PER-SPLIT SUMMARY & CLASS DISTRIBUTION")
    print("=" * 65)

    # นับจำนวนภาพในแต่ละ split และคำนวณเป็นเปอร์เซ็นต์
    split_counts = splits_df["split"].value_counts().to_dict()
    print("Image counts per split:")
    for s in ["train", "val", "test"]:
        cnt = split_counts.get(s, 0)
        pct = (cnt / total_images) * 100
        print(f"  - {s:<6}: {cnt:>5} images ({pct:.2f}%)")

    # สร้างตารางแจกแจงความถี่จำนวนรูปของแต่ละโรค (dx) ในแต่ละ split
    counts_table = pd.crosstab(splits_df["dx"], splits_df["split"])[["train", "val", "test"]]
    counts_table["Total"] = counts_table.sum(axis=1)

    print("\nClass counts per split:")
    print(counts_table.to_string())

    # คำนวณสัดส่วนเป็นเปอร์เซ็นต์ของแต่ละคลาสในแต่ละ split เพื่อดูความสมดุล
    prop_table = pd.crosstab(splits_df["dx"], splits_df["split"], normalize="columns")[["train", "val", "test"]] * 100
    print("\nClass proportions (%) per split:")
    print(prop_table.round(2).to_string())
    print("=" * 65)

    # =========================================================================
    # บันทึกไฟล์ splits.csv ลง Disk
    # =========================================================================
    out_path = Path(output_path)
    # สร้างโฟลเดอร์ปลายทางอัตโนมัติหากยังไม่มี
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # บันทึก DataFrame เป็น CSV โดยไม่บันทึกหมายเลข index
    splits_df.to_csv(out_path, index=False)
    print(f"\nSaved splits to: {out_path.resolve()}\n")

    return splits_df


# บล็อกเริ่มต้นทำงานเมื่อเรียกไฟล์นี้โดยตรงผ่าน Python CLI
if __name__ == "__main__":
    # สร้าง parser สำหรับรับ argument จาก command line
    parser = argparse.ArgumentParser(description="Generate HAM10000 train/val/test splits.")
    # กำหนด option --metadata สำหรับเปลี่ยน path ไฟล์ metadata (default: data/HAM10000_metadata.csv)
    parser.add_argument("--metadata", default="data/HAM10000_metadata.csv", help="Path to HAM10000 metadata CSV")
    # กำหนด option --output สำหรับเปลี่ยน path ไฟล์บันทึก split (default: splits/splits.csv)
    parser.add_argument("--output", default="splits/splits.csv", help="Output path for splits CSV")
    # กำหนด option --seed สำหรับกำหนดค่า random seed (default: 42)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    # แปลง argument ที่รับเข้ามา
    args = parser.parse_args()

    # เรียกใช้ฟังก์ชัน make_splits ด้วยพารามิเตอร์ที่กำหนด
    make_splits(
        metadata_path=args.metadata,
        output_path=args.output,
        seed=args.seed,
    )
