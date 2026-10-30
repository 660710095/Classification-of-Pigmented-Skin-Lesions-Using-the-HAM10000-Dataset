# สรุปผลการทดลอง: ResNet-50 บนชุดข้อมูล HAM10000 (เปรียบเทียบ Exp 1 vs Exp 2)

เอกสารสรุปผลการทดลองการจำแนกรอยโรคผิวหนัง 7 คลาส (Skin Lesion Classification) ด้วยโมเดล **ResNet-50** บนชุดข้อมูล **HAM10000 (10,015 ภาพ)** แบ่งข้อมูลแบบ Stratified Group Split (70% Train / 15% Val / 15% Test) ตาม `lesion_id` เพื่อป้องกัน Data Leakage

---

## 1. ตารางเปรียบเทียบการตั้งค่า (Experiment Configurations)

| การตั้งค่า / ไฮเปอร์พารามิเตอร์ | การทดลองที่ 1 (Baseline 1-Dense) | การทดลองที่ 2 (Proposed 3-Dense Head) |
|---|---|---|
| **Backbone Model** | ResNet-50 (Pretrained ImageNet) | ResNet-50 (Pretrained ImageNet) |
| **Classification Head** | **1 ชั้น**: `Linear(2048 -> 7)` | **3 ชั้น**: `Linear(2048->1024) -> Linear(1024->512) -> Linear(512->7)` + BatchNorm + ReLU + Dropout(0.3) |
| **จำนวนรอบ (Epochs)** | 15 Epochs | **30 Epochs** |
| **Backbone Learning Rate** | `1e-5` | **`3e-5`** (เร่งการสกัดฟีเจอร์) |
| **Head Learning Rate** | `1e-4` | `1e-4` |
| **การแก้ปัญหา Class Imbalance** | Inverse Frequency Weights ใน Loss (ต่างกัน 60 เท่า) | **`WeightedRandomSampler`** (สุ่ม batch สมดุล ไม่บิดเบือน Loss) |
| **Batch Size & Precision** | 64, bfloat16 AMP | 64, bfloat16 AMP |

---

## 2. ผลลัพธ์เปรียบเทียบภาพรวมบน Held-Out Test Set (1,503 ภาพ)

| ตัวชี้วัด (Metrics) | การทดลองที่ 1 (1 ชั้น, 15 Epochs) | การทดลองที่ 2 (3 ชั้น, 30 Epochs) | ผลต่างการพัฒนา (Improvement) |
|---|:---:|:---:|:---:|
| **Overall Accuracy** | 56.09% | **83.10%** | **+27.01%** 🔥 |
| **Macro Average F1** | 0.4859 | **0.7454** | **+25.95%** 🔥 |
| **Weighted Average F1**| 0.6307 | **0.8312** | **+20.05%** 🔥 |
| **Validation Accuracy**| 48.64% | **84.36%** | **+35.72%** 🔥 |

---

## 3. ตารางแจกแจงผลรายคลาสเปรียบเทียบ (Per-Class Performance on Test Set)

| คลาสโรค (dx) | ความหมาย | Exp 1: Recall | Exp 2: Recall | Exp 1: Precision | Exp 2: Precision | Exp 1: F1 | Exp 2: F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **akiec** | ระยะก่อนมะเร็ง | 85.71% | 73.47% | 39.25% | **65.45%** | 0.5385 | **0.6923** |
| **bcc** | มะเร็งผิวหนัง | 79.49% | 79.49% | 55.36% | **67.39%** | 0.6526 | **0.7294** |
| **bkl** | ติ่งเนื้อ/กระเนื้อ | 51.52% | **64.85%** | 65.38% | **70.86%** | 0.5763 | **0.6772** |
| **df** | ก้อนเนื้อไฟโบรมา | 70.59% | **76.47%** | 3.91% ⚠️ | **76.47%** 🚀 | 0.0741 | **0.7647** |
| **mel** | มะเร็งเมลาโนมารุนแรง | 55.95% | 55.95% | 40.87% | **55.95%** | 0.4724 | **0.5595** |
| **nv** | ไฝปกติ (คลาสใหญ่สุด) | 52.64% ⚠️ | **91.44%** 🚀 | 97.24% | **91.90%** | 0.6830 | **0.9167** |
| **vasc** | หลอดเลือดผิดปกติ | 90.48% | 85.71% | 26.03% | **90.00%** 🚀 | 0.4043 | **0.8780** |

---

## 4. วิเคราะห์เจาะลึกสิ่งที่เปลี่ยนไป (In-Depth Technical Analysis)

1. **การแก้ปัญหา Precision พังของคลาสหายากสำเร็จเด็ดขาด**:
   * ในรอบที่ 1 คลาส `df` มี Precision เพียง **3.91%** เนื่องจาก Inverse Class Weights ต่างกัน 60 เท่า โมเดลจึงทายสุ่มสี่สุ่มห้าเพื่อเลี่ยงโทษ
   * ในรอบที่ 2 เมื่อเปลี่ยนมาใช้ **`WeightedRandomSampler`**: ค่า Precision ของ `df` พุ่งกระฉูดขึ้นเป็น **76.47%** และ `vasc` พุ่งขึ้นเป็น **90.00%**
2. **การกอบกู้ Recall ของไฝปกติ (`nv`)**:
   * ไฝปกติ `nv` มีสัดส่วนถึง 67% ของชุดข้อมูล ในรอบแรกทำได้เพียง 52.64% แต่ในรอบที่ 2 กระโดดขึ้นมาเป็น **91.44%** (ตรวจจับถูก 919 จาก 1,005 ภาพ) เป็นสาเหตุสำคัญที่ทำให้ **Overall Accuracy ทะลุ 83.10%**
3. **บทบาทของ Dense 3 ชั้น (`[1024, 512]`)**:
   * การลดมิติแบบเป็นขั้นบันได พร้อมคั่นด้วย `BatchNorm1d`, `ReLU` และ `Dropout(0.3)` ช่วยให้โครงข่ายสกัดความสัมพันธ์เชิงลึก (Non-linear manifold) ได้คมชัดกว่าการโปรเจกต์ 2048 ➔ 7 ชั้นเดียวตรงๆ
4. **ระยะเวลาการเทรน 30 Epochs**:
   * กราฟลู่เข้าอย่างสวยงามที่ประมาณ Epoch 24–28 โดยไม่มีสัญญาณ Overfitting รุนแรงเนื่องจากมี Data Augmentation และ Dropout คุมไว้

---

## 5. ไฟล์และหลักฐานผลลัพธ์ (Artifacts Link)

* กราฟ Loss & Accuracy (PNG): [`resnet50_loss_accuracy.png`](resnet50_loss_accuracy.png)
* กราฟ Confusion Matrix Heatmap (PNG): [`resnet50_confusion_matrix.png`](resnet50_confusion_matrix.png)
* ตารางผลการทดสอบละเอียด (CSV): [`metrics_summary.csv`](metrics_summary.csv)
* รายงานตัวเลข Text Report: [`resnet50_test_report.txt`](resnet50_test_report.txt)
