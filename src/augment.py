"""
src/augment.py
โมดูลสำหรับจัดการ Data Augmentation ของชุดข้อมูล HAM10000
โดยทำการแปลงภาพแบบ On-The-Fly (สุ่มแปลงขณะเทรนในหน่วยความจำเท่านั้น ไม่บันทึกลงดิสก์)
"""

# นำเข้า transforms จาก torchvision เพื่อใช้สร้าง Data Augmentation Pipeline สำหรับรูปภาพ
from torchvision import transforms

# ค่าเฉลี่ย (Mean) ของช่องสี R, G, B มาตรฐานของชุดข้อมูล ImageNet ใช้สำหรับ normalize ค่าสี
IMAGENET_MEAN = [0.485, 0.456, 0.406]
# ค่าเบี่ยงเบนมาตรฐาน (Std) ของช่องสี R, G, B มาตรฐานของชุดข้อมูล ImageNet ใช้สำหรับ normalize ค่าสี
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(img_size: int = 224) -> transforms.Compose:
    """
    สร้าง Transform Pipeline สำหรับข้อมูลชุด Training พร้อม Data Augmentation:
    - สุ่มพลิกภาพแนวนอน (RandomHorizontalFlip)
    - สุ่มพลิกภาพแนวตั้ง (RandomVerticalFlip)
    - สุ่มหมุนภาพ 0-180 องศา (RandomRotation 180)
    - สุ่มครอบภาพและปรับขนาด (RandomResizedCrop) สเกล 80-100%
    - สุ่มปรับเปลี่ยนโทนสี/แสง (ColorJitter: brightness, contrast, saturation, hue)
    - แปลงรูปภาพเป็น Tensor (ToTensor) ค่าพิกเซล 0.0 - 1.0
    - ปรับมาตรฐานค่าสีตาม ImageNet (Normalize)
    """
    return transforms.Compose([
        # สุ่มพลิกรูปภาพในแนวนอน (ซ้าย-ขวา) ด้วยความน่าจะเป็น 50%
        transforms.RandomHorizontalFlip(p=0.5),

        # สุ่มพลิกรูปภาพในแนวตั้ง (บน-ล่าง) ด้วยความน่าจะเป็น 50% (ภาพผิวหนังไม่มีทิศทางตายตัว)
        transforms.RandomVerticalFlip(p=0.5),

        # สุ่มหมุนภาพตามองศาได้สูงสุด 180 องศา (ทั้งตามเข็มและทวนเข็มนาฬิกา)
        transforms.RandomRotation(degrees=180),

        # สุ่มตัดครอบบางส่วนของภาพ (scale ระหว่าง 80% ถึง 100%) แล้วขยายกลับมาขนาด img_size x img_size
        transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0)),

        # สุ่มปรับแต่งสี ความสว่าง และคอนทราสต์เล็กน้อย เพื่อจำลองสภาพแสงของกล้องส่องผิวหนังที่ต่างกัน
        transforms.ColorJitter(
            brightness=0.15,   # สุ่มความสว่าง +/- 15%
            contrast=0.15,     # สุ่มคอนทราสต์ +/- 15%
            saturation=0.15,   # สุ่มความอิ่มตัวสี +/- 15%
            hue=0.03           # สุ่มโทนสี +/- 3%
        ),

        # แปลง PIL Image ขนาด (H, W, C) ค่า 0-255 เป็น PyTorch Tensor ขนาด (C, H, W) ค่าทศนิยม 0.0 - 1.0
        transforms.ToTensor(),

        # ปรับค่าพิกเซลให้มี Mean=0 และ Std=1 ตามมาตรฐาน ImageNet เพื่อให้เข้ากับโมเดล Pre-trained
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(img_size: int = 224) -> transforms.Compose:
    """
    สร้าง Transform Pipeline สำหรับข้อมูลชุด Validation และ Test:
    - ปรับขนาดภาพให้เป็น (img_size, img_size) ด้วยคำสั่ง Resize แบบคงที่
    - แปลงภาพเป็น Tensor ด้วย ToTensor
    - Normalize ค่าสีตามมาตรฐาน ImageNet
    *หมายเหตุ: ไม่มีการใช้ Random Operations ใดๆ เพื่อให้การวัดผลคงที่ แม่นยำ และเที่ยงตรง*
    """
    return transforms.Compose([
        # ปรับขนาดภาพทั้งความกว้างและความสูงให้ตรงกับ img_size ที่โมเดลต้องการโดยไม่สุ่ม
        transforms.Resize((img_size, img_size)),

        # แปลง PIL Image เป็น PyTorch Float Tensor ช่วงค่า [0.0, 1.0]
        transforms.ToTensor(),

        # ปรับ Normalize ค่าสีด้วย ImageNet Mean และ Std ให้ตรงกับตอนเทรน
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
