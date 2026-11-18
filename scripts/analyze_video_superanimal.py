"""
Prueba del modelo preentrenado SuperAnimal-Quadruped (39 keypoints) sobre un video.
No requiere entrenar ni etiquetar. La primera vez descarga los pesos de Hugging Face.

Uso (desde la raiz del repo, con el entorno pose_canina activo):
    python scripts/analyze_video_superanimal.py video1.mp4 [video2.mp4 ...]
"""
import sys
from pathlib import Path

import deeplabcut
import pandas as pd

def analizar(video: Path):
    dest = video.parent / f"{video.stem}_superanimal"
    if list(dest.glob("*.h5")):
        print(f"Ya analizado, se omite: {video.name}")
        return dest
    dest.mkdir(exist_ok=True)

    deeplabcut.video_inference_superanimal(
        [str(video)],
        superanimal_name="superanimal_quadruped",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_resnet50_fpn_v2",
        video_extensions=".mp4",
        dest_folder=str(dest),
        pcutoff=0.6,
        batch_size=8,
        detector_batch_size=1,
        max_individuals=1,
        plot_bboxes=False,
        create_labeled_video=True,
    )
    return dest


def resumen(dest: Path):
    """Confianza media y % de frames con confianza > 0.6 por keypoint."""
    h5 = list(dest.glob("*.h5"))
    if not h5:
        return
    d = pd.read_hdf(h5[0])
    lik = d.xs("likelihood", axis=1, level=-1)
    if lik.columns.nlevels > 1:
        lik = lik.T.groupby(level=-1).mean().T  # un solo individuo
    print(f"\n{len(d)} frames  ->  {h5[0].name}")
    for k in lik.columns:
        print(f"{str(k):22s} media={lik[k].mean():.2f}  >0.6: {(lik[k] > 0.6).mean()*100:3.0f}%")


if __name__ == "__main__":
    # Acepta uno o varios videos: se procesan en orden y se omiten los ya analizados.
    for arg in sys.argv[1:]:
        video = Path(arg).resolve()
        resumen(analizar(video))
