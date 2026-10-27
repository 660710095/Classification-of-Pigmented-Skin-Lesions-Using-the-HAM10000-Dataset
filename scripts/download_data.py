#!/usr/bin/env python3
"""
scripts/download_data.py
สคริปต์ตัวช่วยดาวน์โหลดชุดข้อมูล HAM10000 ผ่าน kagglehub
และสร้าง Symlink เชื่อมโยงเข้าโฟลเดอร์ data/ โดยอัตโนมัติ
"""

# นำเข้าโมดูล os สำหรับการทำงานร่วมกับระบบปฏิบัติการ
import os
# นำเข้า Path จาก pathlib สำหรับจัดการ file path
from pathlib import Path
# นำเข้าไลบรารี kagglehub สำหรับดาวน์โหลดชุดข้อมูลจาก Kaggle
import kagglehub


def setup_data():
    """
    ฟังก์ชันดาวน์โหลดชุดข้อมูลและสร้าง Symlink เชื่อมต่อมายังโฟลเดอร์ data/
    """
    # ระบุตำแหน่งโฟลเดอร์รากของโปรเจกต์ (term_Project)
    project_root = Path(__file__).resolve().parent.parent
    # ระบุตำแหน่งโฟลเดอร์ data ของโปรเจกต์
    data_dir = project_root / "data"
    # สร้างโฟลเดอร์ data/ หากยังไม่มี
    data_dir.mkdir(parents=True, exist_ok=True)

    print("กำลังดาวน์โหลดชุดข้อมูล HAM10000 ผ่าน kagglehub...")
    # ดาวน์โหลดเวอร์ชันล่าสุดของ HAM10000 จาก Kaggle และส่งกลับพาธที่เก็บไฟล์
    download_path = Path(kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000"))
    print("ดาวน์โหลดสำเร็จ! ที่อยู่ไฟล์ต้นฉบับ:", download_path)

    # กำหนดรายการโฟลเดอร์และไฟล์ที่ต้องการเชื่อมโยง พร้อมรายชื่อตัวพิมพ์เล็ก/ใหญ่ที่อาจพบ
    targets = {
        "HAM10000_images_part_1": ["HAM10000_images_part_1", "ham10000_images_part_1"],
        "HAM10000_images_part_2": ["HAM10000_images_part_2", "ham10000_images_part_2"],
        "HAM10000_metadata.csv": ["HAM10000_metadata.csv"],
    }

    # วนลูปสร้าง Symlink ไปยังแต่ละโฟลเดอร์/ไฟล์
    for target_name, candidates in targets.items():
        # กำหนดที่อยู่ปลายทางในโฟลเดอร์ data/
        dest = data_dir / target_name
        
        # หากมีไฟล์หรือ symlink อยู่แล้ว ให้ข้ามไปไม่ต้องสร้างซ้ำ
        if dest.exists() or dest.is_symlink():
            print(f"พบไฟล์/โฟลเดอร์ปลายทางแล้ว: {dest}")
            continue

        # ค้นหาว่าไฟล์ต้นฉบับตรงกับชื่อใดใน candidates
        src = None
        for candidate in candidates:
            cand_path = download_path / candidate
            if cand_path.exists():
                src = cand_path
                break

        # หากไม่พบไฟล์ต้นฉบับเลย ให้แจ้งข้อผิดพลาด
        if src is None:
            raise FileNotFoundError(f"ไม่พบ {target_name} ในโฟลเดอร์ {download_path}")

        # สร้าง Symlink จาก dest เชื่อมโยงไปยัง src เพื่อไม่ให้เปลืองพื้นที่ดิสก์ซ้ำซ้อน
        print(f"สร้าง Symlink: {dest} -> {src}")
        dest.symlink_to(src)

    # แสดงรายการทั้งหมดที่เชื่อมโยงเรียบร้อยใน data/
    print("ตั้งค่าข้อมูลเสร็จสมบูรณ์! รายการใน data/:")
    for item in data_dir.iterdir():
        print(f"  {item.name} -> {item.resolve()}")


# สั่งให้ทำงานเมื่อรันไฟล์นี้โดยตรง
if __name__ == "__main__":
    setup_data()
