import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent          # scripts/
REPO_ROOT = THIS_DIR.parent                           # raiz del repo

sys.path.append(str(THIS_DIR))
sys.path.append(str(REPO_ROOT / "data" / "loaders"))

from convert_to_dlc_format import convert_to_dlc_format
from stanfordextra import load_stanfordextra

BODYPARTS = [
    "left_front_paw", "left_front_knee", "left_front_top",
    "left_rear_paw", "left_rear_knee", "left_rear_top",
    "right_front_paw", "right_front_knee", "right_front_top",
    "right_rear_paw", "right_rear_knee", "right_rear_top",
    "tail_start", "tail_end",
    "left_ear_base", "right_ear_base",
    "nose", "chin",
    "left_ear_tip", "right_ear_tip",
    "left_eye", "right_eye",
    "withers", "throat",
]

JSON_PATH = r"C:\Users\usuario\Desktop\UADY\SemestreAgostoDiciembre2026\Practicas\DataSetStanford\StanfordExtra_V12\StanfordExtra_v12.json"
SOURCE_IMAGES_DIR = REPO_ROOT / "data" / "images" / "Images"
DEST_DIR = REPO_ROOT / "dlc_project" / "labeled-data" / "StanfordExtra"
SCORER = "Fernando Loza"

if __name__ == "__main__":
    datos = load_stanfordextra(JSON_PATH, BODYPARTS)
    print(f"Cargadas {len(datos)} anotaciones del JSON.")

    df = convert_to_dlc_format(
        loaded_data=datos,
        bodypart_names=BODYPARTS,
        scorer=SCORER,
        source_images_dir=SOURCE_IMAGES_DIR,
        dest_dir=DEST_DIR,
    )
    print(df.shape)