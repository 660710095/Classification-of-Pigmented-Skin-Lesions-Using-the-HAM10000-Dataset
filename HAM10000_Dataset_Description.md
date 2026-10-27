# HAM10000 (Skin Cancer MNIST) – รายละเอียดชุดข้อมูล

## ภาพรวม

| หัวข้อ | รายละเอียด |
|---|---|
| ชื่อเต็ม | Human Against Machine with 10000 training images |
| ประเภทข้อมูล | ภาพถ่ายระยะใกล้ของรอยโรคผิวหนังที่มีสี (pigmented lesions) ด้วยกล้อง dermatoscope |
| จำนวนรูป | 10,015 รูป |
| จำนวนคลาส | **7 คลาส** |
| ผู้สร้าง | Tschandl, Rosendahl, Kittler (2018), Medical University of Vienna |
| แหล่งที่มาของภาพ | ภาควิชาผิวหนัง Medical University of Vienna (ออสเตรีย) และคลินิกมะเร็งผิวหนังของ Cliff Rosendahl (ควีนส์แลนด์, ออสเตรเลีย) |
| Kaggle | https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 |
| Harvard Dataverse | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T |

---

## 1. คลาสทั้งหมด (7 คลาส)

คลาสอยู่ในคอลัมน์ `dx` ของไฟล์ metadata

| # | ชื่อย่อ (`dx`) | ชื่อเต็ม | ความหมาย | กลุ่มความร้ายแรง | จำนวนรูป | สัดส่วน |
|---|---|---|---|---|---|---|
| 1 | **nv** | Melanocytic nevi | ไฝธรรมดา เกิดจากเซลล์สร้างเม็ดสี ไม่เป็นอันตราย | Benign (ไม่ร้ายแรง) | 6,705 | 66.95% |
| 2 | **mel** | Melanoma | มะเร็งผิวหนังชนิดร้ายแรงที่สุด เกิดจากเซลล์สร้างเม็ดสีกลายพันธุ์ | Malignant (มะเร็ง) | 1,113 | 11.11% |
| 3 | **bkl** | Benign keratosis-like lesions | รอยโรคคล้ายเคราตินชนิดไม่ร้ายแรง รวม solar lentigines (กระแดด), seborrheic keratoses (กระเนื้อ/รอยกระจากวัยชรา) และ lichen-planus like keratoses | Benign | 1,099 | 10.97% |
| 4 | **bcc** | Basal cell carcinoma | มะเร็งผิวหนังเซลล์ฐาน พบบ่อย โตช้า แพร่กระจายน้อย | Malignant | 514 | 5.13% |
| 5 | **akiec** | Actinic keratoses and intraepithelial carcinoma / Bowen's disease | รอยโรคก่อนเป็นมะเร็งจากแสงแดด (actinic keratoses) และมะเร็งระยะเริ่มต้นที่อยู่เฉพาะชั้นผิว (Bowen's disease) | Pre-malignant (ก่อนเป็นมะเร็ง) | 327 | 3.27% |
| 6 | **vasc** | Vascular lesions | รอยโรคหลอดเลือด เช่น angiomas, angiokeratomas, pyogenic granulomas และ hemorrhage (เลือดออกใต้ผิว) | Benign | 142 | 1.42% |
| 7 | **df** | Dermatofibroma | ก้อนเนื้อเยื่อเกี่ยวพันในผิวหนัง ไม่เป็นอันตราย | Benign | 115 | 1.15% |
| | | **รวม** | | | **10,015** | 100% |

**หมายเหตุ:** ชุดข้อมูลมีความไม่สมดุลของคลาสสูงมาก nv มากกว่า df ประมาณ 58 เท่า

---

## 2. โครงสร้างไฟล์: "สอง part" คืออะไร

รูปภาพทั้ง 10,015 รูปถูกแบ่งเก็บเป็น **2 โฟลเดอร์** บน Kaggle:

| โฟลเดอร์ | เนื้อหา |
|---|---|
| `HAM10000_images_part_1/` | รูป JPG ส่วนแรก (ประมาณครึ่งหนึ่ง) |
| `HAM10000_images_part_2/` | รูป JPG ส่วนที่เหลือ |

**สิ่งที่ต้องเข้าใจ:**
- ทั้งสอง part เป็นเพียง **การแบ่งโฟลเดอร์เพื่อความสะดวกในการอัปโหลด/ดาวน์โหลด** ไม่ใช่การแบ่ง train/test และไม่ใช่การแบ่งตามคลาส ทุกคลาสปะปนอยู่ในทั้งสองโฟลเดอร์
- ชื่อไฟล์ใช้รูปแบบ `ISIC_xxxxxxx.jpg` ตรงกับคอลัมน์ `image_id` ใน metadata
- เวลาโหลดข้อมูลต้อง **รวม path ของทั้งสองโฟลเดอร์** เพื่อหารูปให้ครบ
- จำนวนรูปที่แน่นอนในแต่ละ part ควรนับจากไฟล์จริง (ผมไม่ได้ตรวจจากไฟล์)

---

## 3. ความหมายของแต่ละไฟล์

### 3.1 ไฟล์รูปต้นฉบับ

| ไฟล์ / โฟลเดอร์ | ความหมาย |
|---|---|
| `HAM10000_images_part_1/` | รูป dermatoscope ต้นฉบับ (JPG) ส่วนที่ 1 ขนาดประมาณ 600×450 พิกเซล |
| `HAM10000_images_part_2/` | รูป dermatoscope ต้นฉบับ (JPG) ส่วนที่ 2 |

### 3.2 ไฟล์ metadata

| ไฟล์ | ความหมาย |
|---|---|
| `HAM10000_metadata.csv` | ตารางข้อมูลประกอบของทุกรูป (10,015 แถว) ใช้เชื่อมรูปกับ label และข้อมูลผู้ป่วย |

คอลัมน์ใน `HAM10000_metadata.csv`:

| คอลัมน์ | ความหมาย | หมายเหตุ |
|---|---|---|
| `lesion_id` | รหัสรอยโรค | รอยโรคหนึ่งอาจมีหลายรูป จึงต้องใช้แบ่ง train/val/test เพื่อกัน data leakage |
| `image_id` | รหัสรูปภาพ | ตรงกับชื่อไฟล์ `.jpg` |
| `dx` | การวินิจฉัย (label 7 คลาส) | ตัวแปรเป้าหมาย |
| `dx_type` | วิธียืนยันการวินิจฉัย | `histo` = ผลตัดชิ้นเนื้อ (มากกว่า 50% ของข้อมูล), `follow_up` = ตรวจติดตามผล, `consensus` = มติผู้เชี่ยวชาญ, `confocal` = ยืนยันด้วยกล้อง in-vivo confocal microscopy |
| `age` | อายุผู้ป่วย | มีค่าว่างบางส่วน |
| `sex` | เพศ | male / female / unknown |
| `localization` | ตำแหน่งของรอยโรคบนร่างกาย | เช่น back, face, lower extremity, trunk |

### 3.3 ไฟล์รูปที่ย่อเป็นตัวเลข (ใช้ทดลองเร็ว)

ไฟล์กลุ่ม `hmnist_*.csv` เป็นรูปที่ย่อขนาดและเก็บเป็นค่าพิกเซลในตาราง CSV โดยแต่ละแถวคือ 1 รูป คอลัมน์สุดท้าย `label` เป็นเลขคลาส (0-6)

| ไฟล์ | ความหมาย |
|---|---|
| `hmnist_28_28_RGB.csv` | รูปขนาด 28×28 สี (RGB) พิกเซลละ 3 ค่า รวม 2,352 คอลัมน์ + label |
| `hmnist_28_28_L.csv` | รูปขนาด 28×28 ขาวดำ (grayscale) 784 คอลัมน์ + label |
| `hmnist_8_8_RGB.csv` | รูปขนาด 8×8 สี 192 คอลัมน์ + label |
| `hmnist_8_8_L.csv` | รูปขนาด 8×8 ขาวดำ 64 คอลัมน์ + label |

การจับคู่เลข label กับคลาส (ตามที่เห็นในโค้ดตัวอย่างบน Kaggle ควรยืนยันกับ notebook ของดาต้าเซทอีกที):

| label | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| คลาส | akiec | bcc | bkl | df | nv | vasc | mel |

**ข้อแนะนำ:** ไฟล์ `hmnist_*` เหมาะกับการทดลองเบื้องต้นหรือสอนเท่านั้น รูปเล็กเกินไปจนสูญเสียรายละเอียดที่ใช้แยกโรค โปรเจคหลักควรใช้รูปต้นฉบับจากสองโฟลเดอร์ข้างต้น

---

## 4. ข้อควรรู้เกี่ยวกับชุดข้อมูล

- **Test set ทางการไม่เปิดเผย:** ผู้สร้างระบุว่าผลงานตีพิมพ์ควรประเมินบน test set ของเซิร์ฟเวอร์ ISIC (https://challenge2018.isic-archive.com) การแบ่ง train/val/test เองจึงเทียบกับ leaderboard ทางการโดยตรงไม่ได้
- **หนึ่งรอยโรคอาจมีหลายรูป:** แบ่งข้อมูลตาม `lesion_id` ไม่ใช่ตามรูป
- **ความหลากหลายของประชากร:** ผู้ป่วยส่วนใหญ่มาจากออสเตรียและออสเตรเลีย
- **ใช้เพื่อการศึกษา/วิจัยเท่านั้น** ไม่ใช่เครื่องมือวินิจฉัยจริง

## อ้างอิง

1. Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci. Data 5, 180161 (2018). doi:10.1038/sdata.2018.161
2. Codella, N. et al. *Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC).* arXiv:1902.03368
