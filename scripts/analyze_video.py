"""
Corre el modelo entrenado sobre un video y genera el video con los keypoints dibujados.

Uso (desde la raiz del repo, con el entorno pose_canina activo):
    python scripts/analyze_video.py
    python scripts/analyze_video.py ruta/a/otro_video.mp4
"""
import sys
from pathlib import Path

import deeplabcut

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "dlc_project_12breeds" / "config.yaml"
DEFAULT_VIDEO = REPO_ROOT.parent / "perroPaseando.mp4"

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_VIDEO)

    deeplabcut.analyze_videos(str(CONFIG), [video], shuffle=1, save_as_csv=True)
    deeplabcut.create_labeled_video(str(CONFIG), [video], shuffle=1, draw_skeleton=False)
