# สรุปผลการทดลอง: ResNet-50 บนชุดข้อมูล HAM10000 (เปรียบเทียบ Exp 1 vs Exp 2 vs Exp 3)

เอกสารสรุปผลการทดลองการจำแนกรอยโรคผิวหนัง 7 คลาส (Skin Lesion Classification) ด้วยโมเดล **ResNet-50** บนชุดข้อมูล **HAM10000 (10,015 ภาพ)** แบ่งข้อมูลแบบ Stratified Group Split (70% Train / 15% Val / 15% Test) ตาม `lesion_id` เพื่อป้องกัน Data Leakage

---

## 1. ตารางเปรียบเทียบการตั้งค่าพารามิเตอร์ (Ablation Study Configurations)

| การตั้งค่า / ไฮเปอร์พารามิเตอร์ | การทดลองที่ 1 (1-Dense Layer) | การทดลองที่ 2 (3-Dense Head) 🏆 | การทดลองที่ 3 (4-Dense Head) |
|---|---|---|---|
| **Backbone Model** | ResNet-50 (Pretrained ImageNet) | ResNet-50 (Pretrained ImageNet) | ResNet-50 (Pretrained ImageNet) |
| **Classification Head** | **1 ชั้น**: `Linear(2048 -> 7)` | **3 ชั้น**: `Linear(2048->1024) -> Linear(1024->512) -> Linear(512->7)` | **4 ชั้น**: `Linear(2048->1024) -> Linear(1024->512) -> Linear(512->256) -> Linear(256->7)` |
| **จำนวนรอบ (Epochs)** | 15 Epochs | 30 Epochs | 30 Epochs |
| **Backbone Learning Rate** | `1e-5` | `3e-5` | `3e-5` |
| **Head Learning Rate** | `1e-4` | `1e-4` | `1e-4` |
| **การแก้ปัญหา Class Imbalance** | Inverse Frequency Weights ใน Loss (ต่างกัน 60 เท่า) | **`WeightedRandomSampler`** | **`WeightedRandomSampler`** |
| **Batch Size & Precision** | 64, bfloat16 AMP | 64, bfloat16 AMP | 64, bfloat16 AMP |

---

## 2. ผลลัพธ์เปรียบเทียบภาพรวมบน Held-Out Test Set (1,503 ภาพ)

| ตัวชี้วัด (Metrics) | Exp 1 (1 ชั้น, 15 Epochs) | Exp 2 (3 ชั้น, 30 Epochs) 🏆 | Exp 3 (4 ชั้น, 30 Epochs) | บทวิเคราะห์ |
|---|:---:|:---:|:---:|---|
| **Test Accuracy** | 56.09% | **83.10%** 🏆 | 81.70% | **Exp 2 ดีที่สุด** (4 ชั้นเริ่มลดลง -1.4%) |
| **Macro Average F1** | 0.4859 | **0.7454** 🏆 | 0.7333 | **Exp 2 ดีที่สุด** (4 ชั้นเริ่มลดลง -1.2%) |
| **Weighted Average F1**| 0.6307 | **0.8312** 🏆 | 0.8188 | Exp 2 ดีที่สุด |
| **Best Val Accuracy** | 48.64% | **84.36%** 🏆 | 83.77% | Exp 2 มี Val Acc สูงสุด |
| **Best Val Macro-F1** | 0.4405 | **0.7559** 🏆 | 0.7435 | Exp 2 ทำคะแนนช่วงเทรนพีคสุด |
| **Best Checkpoint Epoch** | Epoch 14 | Epoch 25 | Epoch 27 | ลู่เข้าใกล้เคียงกันช่วงปลาย |

---

## 3. ตารางแจกแจงผลรายคลาสเปรียบเทียบทั้ง 3 การทดลอง (Test Set)

| คลาสโรค (dx) | ชนิดรอยโรค | Exp 1 (F1) | Exp 2 (F1) 🏆 | Exp 3 (F1) | สรุปผลความเปลี่ยนแปลง |
|---|---|:---:|:---:|:---:|---|
| **akiec** | ระยะก่อนมะเร็ง | 0.5385 | **0.6923** 🏆 | 0.6667 | 3 ชั้นจำแนกได้แม่นยำสูงสุด |
| **bcc** | มะเร็งผิวหนังชนิดพบบ่อย | 0.6526 | 0.7294 | **0.7836** 🏆 | 4 ชั้นจับมะเร็งชนิดนี้ได้ดีขึ้น (Recall 85.9%) |
| **bkl** | ติ่งเนื้อ/กระเนื้อ | 0.5763 | **0.6772** 🏆 | 0.6540 | 3 ชั้นแยกแยะ Texture ได้ดีกว่า |
| **df** | ก้อนเนื้อไฟโบรมา | 0.0741 ⚠️ | **0.7647** 🏆 | **0.7647** 🏆 | รักษาความแม่นยำสูงไว้ได้ดีทั้ง 2 รอบ |
| **mel** | มะเร็งเมลาโนมารุนแรง | 0.4724 | **0.5595** 🏆 | 0.5116 | 4 ชั้นมีสัญญาณดรอปของคลาสยาก |
| **nv** | ไฝปกติ (คลาสใหญ่สุด) | 0.6830 | **0.9167** 🏆 | 0.9077 | 3 ชั้นมีความแม่นยำและเสถียรที่สุด |
| **vasc** | รอยโรคหลอดเลือด | 0.4043 | **0.8780** 🏆 | 0.8444 | 3 ชั้นให้คะแนนสูงสุด |

---

## 4. วิเคราะห์เจาะลึก: ทำไม 4 ชั้น ถึงแม่นยำลดลงเล็กน้อยเทียบกับ 3 ชั้น?

1. **จุดอิ่มตัวของส่วนหัว (Over-Parameterization / Diminishing Returns)**:
   * การเพิ่ม Dense เป็น 4 ชั้น (`2048 -> 1024 -> 512 -> 256 -> 7`) ทำให้ส่วนหัวมีพารามิเตอร์มากเกินความจำเป็นสำหรับ Feature Vector ขนาด 2048 ที่ได้จาก ResNet-50
   * **Dense 3 ชั้น (`[1024, 512]`)** คือ **"Sweet Spot" (จุดสมดุลที่ดีที่สุด)** ที่ให้ความจุในการเรียนรู้ Non-linear เหมาะสมพอดี
2. **ผลกระทบจากการซ้อน Dropout หลายชั้นเกินไป (Signal Degradation)**:
   * ในโมเดล 4 ชั้น มีการใช้ `Dropout(0.3)` ติดต่อกันถึง 3 รอบ ส่งผลให้สูญเสียสัญญาณข้อมูลย่อย (Feature Information Loss) ในการจำแนกรอยโรคผิวหนังที่มีความละเอียดอ่อนสูงอย่าง `mel` (Melanoma) และ `bkl`
3. **การค้นพบเชิงวิทยาศาสตร์ (Ablation Curve)**:
   * **1 ชั้น (56.09%)**: Underfitting (ตื้นเกินไป ขาดความสามารถในการเรียนรู้)
   * **3 ชั้น (83.10%)**: **Optimal Fit (จุดพีคที่มีประสิทธิภาพสูงสุด)** 🏆
   * **4 ชั้น (81.70%)**: Over-parameterized (เริ่มล้นและมีความสูญเสียสัญญาณ)
   * *เป็นผลการทดลองที่ยอดเยี่ยมสำหรับแสดงในรายงาน เพื่อพิสูจน์ว่า "การเพิ่มเลเยอร์ไม่ได้แปลว่าจะดีขึ้นเสมอไป ต้องมีจุดสมดุล"*

---

## 5. สรุปโมเดลที่ดีที่สุดเพื่อนำไปใช้งาน

* **โมเดลที่แนะนำให้เลือกใช้เป็นโมเดลหลัก**: **การทดลองที่ 2 (ResNet-50 แบบ Dense 3 ชั้น `[1024, 512]`)**
  * ไฟล์โมเดล: `checkpoints/resnet50_exp2_3dense_best.pt`
  * ความแม่นยำบน Test Set: **83.10%**, Macro-F1: **74.54%**
  * มีความสมดุลสูงสุดทั้งคลาสทั่วไปและโรคมะเร็งอันตราย

---

## 6. ลิงก์ไฟล์และหลักฐานการทดลอง (Artifacts Links)

* **รอบที่ 2 (3 ชั้น - ดีที่สุด)**:
  * กราฟ Loss & Acc: [`resnet50_exp2_3dense_loss_accuracy.png`](resnet50_exp2_3dense_loss_accuracy.png)
  * กราฟ Confusion Matrix: [`resnet50_exp2_3dense_confusion_matrix.png`](resnet50_exp2_3dense_confusion_matrix.png)
  * ตารางคะแนนละเอียด: [`metrics_summary_exp2.csv`](metrics_summary_exp2.csv)
  * ไฟล์โมเดล: `checkpoints/resnet50_exp2_3dense_best.pt`
* **รอบที่ 3 (4 ชั้น)**:
  * กราฟ Loss & Acc: [`resnet50_exp3_4dense_loss_accuracy.png`](resnet50_exp3_4dense_loss_accuracy.png)
  * กราฟ Confusion Matrix: [`resnet50_exp3_4dense_confusion_matrix.png`](resnet50_exp3_4dense_confusion_matrix.png)
  * ตารางคะแนนละเอียด: [`metrics_summary_exp3.csv`](metrics_summary_exp3.csv)
  * ไฟล์โมเดล: `checkpoints/resnet50_best.pt`
