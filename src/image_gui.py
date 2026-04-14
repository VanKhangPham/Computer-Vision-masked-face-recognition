from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app import load_mask_model, run_gui
from src.vision_utils import configure_output_encoding, load_face_detector


def main() -> None:
    configure_output_encoding()

    print("╔══════════════════════════════════════════╗")
    print("║ FACE MASK – IMAGE GUI FALLBACK DEMO     ║")
    print("╚══════════════════════════════════════════╝\n")

    face_net = load_face_detector()
    print("  Face detector sẵn sàng (OpenCV DNN – Caffe SSD)")
    mask_model = load_mask_model()
    run_gui(face_net, mask_model)


if __name__ == "__main__":
    main()
