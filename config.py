
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List


# 
# ĐƯỜNG DẪN DỰ ÁN
# 
ROOT_DIR      = Path(__file__).parent          # Thư mục gốc dự án
DATA_RAW      = ROOT_DIR / "data" / "raw"
DATA_PROC     = ROOT_DIR / "data" / "processed"
MODELS_DIR    = ROOT_DIR / "models"
RESULTS_DIR   = ROOT_DIR / "results"
PLOTS_DIR     = RESULTS_DIR / "plots"
LOGS_DIR      = RESULTS_DIR / "logs"
SCREENSHOTS   = RESULTS_DIR / "screenshots"
SRC_DIR       = ROOT_DIR / "src"

# Tự tạo nếu chưa có
for _d in [DATA_RAW, DATA_PROC, MODELS_DIR, PLOTS_DIR, LOGS_DIR, SCREENSHOTS]:
    _d.mkdir(parents=True, exist_ok=True)


# 
# CẤU HÌNH CHÍNH (dataclass – gõ có autocomplete trong VSCode)
# 
@dataclass
class Config:
    #  Dữ liệu 
    CLASSES: List[str]       = field(default_factory=lambda: ["with_mask", "without_mask"])
    IMAGE_SIZE: Tuple        = (224, 224)
    BATCH_SIZE: int          = 32
    TRAIN_RATIO: float       = 0.80
    VAL_RATIO: float         = 0.10
    TEST_RATIO: float        = 0.10
    RANDOM_SEED: int         = 42

    #  Model 
    BACKBONE: str            = "MobileNetV2"     # Thay bằng "ResNet50" nếu muốn
    PRETRAINED_WEIGHTS: str  = "imagenet"
    DROPOUT_1: float         = 0.5
    DROPOUT_2: float         = 0.3
    DENSE_UNITS: int         = 128

    # Training 
    INIT_LR: float           = 1e-4              # Learning rate giai đoạn 1
    FINE_TUNE_LR: float      = 1e-5              # Learning rate fine-tune
    EPOCHS_PHASE1: int       = 20
    EPOCHS_PHASE2: int       = 10
    FINE_TUNE_AT: int        = 100               # Unfreeze từ layer này trở đi
    EARLY_STOP_PATIENCE: int = 5

    #  Inference (App) 
    FACE_CONFIDENCE: float   = 0.5               # Ngưỡng phát hiện khuôn mặt
    MASK_THRESHOLD: float    = 0.5               # Ngưỡng phân loại mask
    CAMERA_ID: int           = 0                 # Index webcam (0 = mặc định)
    CAMERA_WIDTH: int        = 1280
    CAMERA_HEIGHT: int       = 720

    # Đường dẫn file model 
    FACE_PROTOTXT: str       = str(MODELS_DIR / "deploy.prototxt")
    FACE_WEIGHTS: str        = str(MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel")
    MASK_MODEL: str          = str(MODELS_DIR / "mask_detector_final.keras")


# Singleton – import ở bất kỳ file nào
CFG = Config()


# 
# KIỂM TRA NHANH KHI CHẠY TRỰC TIẾP
# 
if __name__ == "__main__":
    print("=" * 50)
    print("  CẤU HÌNH DỰ ÁN FACE MASK DETECTION")
    print("=" * 50)
    print(f"  ROOT_DIR   : {ROOT_DIR}")
    print(f"  DATA_RAW   : {DATA_RAW}")
    print(f"  DATA_PROC  : {DATA_PROC}")
    print(f"  MODELS_DIR : {MODELS_DIR}")
    print(f"  RESULTS_DIR: {RESULTS_DIR}")
    print()
    print(f"  IMAGE_SIZE : {CFG.IMAGE_SIZE}")
    print(f"  BATCH_SIZE : {CFG.BATCH_SIZE}")
    print(f"  BACKBONE   : {CFG.BACKBONE}")
    print(f"  CLASSES    : {CFG.CLASSES}")
    print("=" * 50)
    print("Config OK!")
