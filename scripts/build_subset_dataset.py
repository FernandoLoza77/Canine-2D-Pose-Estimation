"""
Construye un proyecto DeepLabCut LIVIANO con solo algunas razas de StanfordExtra.

Uso (desde la raiz del repo, con el entorno pose_canina activo):
    python scripts/build_subset_dataset.py
    python scripts/build_subset_dataset.py --breeds German_shepherd Siberian_husky beagle

Efecto:
    <repo>/<project_name>/config.yaml                 (copia de dlc_project/config.yaml,
                                                      con project_path y video_sets actualizados)
    <repo>/<project_name>/labeled-data/StanfordExtra/ (imagenes de esas razas +
                                                      CollectedData_<scorer>.h5/.csv)
"""
import argparse
import re
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent          # scripts/
REPO_ROOT = THIS_DIR.parent                           # raiz del repo
PRACTICAS = REPO_ROOT.parent                          # carpeta que contiene el repo

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

DEFAULT_JSON = PRACTICAS / "DataSetStanford" / "StanfordExtra_V12" / "StanfordExtra_v12.json"
DEFAULT_WIN_ROOT = r"C:\Users\usuario\Desktop\UADY\SemestreAgostoDiciembre2026\Practicas\Canine-2D-Pose-Estimation"
SCORER = "Fernando Loza"


def breed_of(image_path):
    """'n02106662-German_shepherd/n02106662_123.jpg' -> 'german_shepherd'"""
    folder = Path(image_path).parts[0]
    return folder.split("-", 1)[1].lower()


def write_config(template_path, dest_path, project_path, video_path):
    """Copia config.yaml reescribiendo solo project_path y la clave de video_sets."""
    lines = Path(template_path).read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if line.startswith("project_path:"):
            line = f"project_path: {project_path}"
        elif line.startswith("  ? "):
            line = f"  ? {video_path}"
        out.append(line)
    Path(dest_path).write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--breeds", nargs="+",
                        default=["German_shepherd", "Siberian_husky", "beagle"])
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--project-name", default="dlc_project_3breeds")
    parser.add_argument("--win-root", default=DEFAULT_WIN_ROOT,
                        help="Ruta del repo tal como la ve Windows (para project_path)")
    args = parser.parse_args()

    wanted = {b.lower() for b in args.breeds}
    todos = load_stanfordextra(args.json, BODYPARTS)
    subset = [d for d in todos if breed_of(d["image_path"]) in wanted]

    # Cada raza pedida debe existir en el JSON
    encontradas = {breed_of(d["image_path"]) for d in subset}
    faltan = wanted - encontradas
    if faltan:
        raise SystemExit(f"Razas no encontradas en el JSON: {sorted(faltan)}")

    print(f"{len(todos)} anotaciones en total -> {len(subset)} en las razas elegidas:")
    for b in sorted(encontradas):
        print(f"  {b}: {sum(1 for d in subset if breed_of(d['image_path']) == b)}")

    project_dir = REPO_ROOT / args.project_name
    convert_to_dlc_format(
        loaded_data=subset,
        bodypart_names=BODYPARTS,
        scorer=SCORER,
        source_images_dir=REPO_ROOT / "data" / "images" / "Images",
        dest_dir=project_dir / "labeled-data" / "StanfordExtra",
    )

    win_project = f"{args.win_root}\\{args.project_name}"
    write_config(
        template_path=REPO_ROOT / "dlc_project" / "config.yaml",
        dest_path=project_dir / "config.yaml",
        project_path=win_project,
        video_path=f"{win_project}\\videos\\perroPaseando.mp4",
    )
    print(f"config.yaml escrito en {project_dir / 'config.yaml'}")
