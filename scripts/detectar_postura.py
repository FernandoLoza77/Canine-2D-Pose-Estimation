"""
Herramienta para veterinaria: de un video de un perro a su postura en el tiempo
(acostado / sentado / parado).

Pasos internos:
  1. Reduce el video a pocos frames por segundo (la postura cambia lento).
  2. Detecta los keypoints con el modelo preentrenado SuperAnimal-Quadruped.
  3. Clasifica la postura de cada frame con reglas geometricas (classify_posture.py).
  4. Guarda un CSV con la linea de tiempo y muestra un resumen.

Uso (desde la raiz del repo, con el entorno activo):
    python scripts/detectar_postura.py ruta/al/video.mp4
    python scripts/detectar_postura.py ruta/al/video.mp4 --fps 1 --salida resultados

Requiere vista lateral y un solo perro en cuadro (ver README).
"""
import argparse
from pathlib import Path

import cv2
import pandas as pd

from classify_posture import clasificar, medidas_por_frame


def submuestrear(video: Path, destino: Path, fps_salida: float) -> float:
    """Guarda en `destino` una copia del video con `fps_salida` frames por segundo.
    Devuelve los segundos que dura cada frame del video reducido."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    paso = max(1, round(fps / fps_salida))
    fps_real = fps / paso

    out = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"), fps_real, (ancho, alto))
    i = guardados = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % paso == 0:
            out.write(frame)
            guardados += 1
        i += 1
    cap.release()
    out.release()
    if guardados == 0:
        raise SystemExit("El video no tiene frames legibles.")
    return 1.0 / fps_real


def postura_desde_h5(h5: Path, seg_por_frame: float) -> pd.DataFrame:
    """Linea de tiempo de posturas a partir del .h5 de SuperAnimal."""
    d = pd.read_hdf(h5)
    m = medidas_por_frame(d)
    m["postura"] = clasificar(m)
    # Un frame dudoso hereda la postura del anterior (la postura no cambia en un frame)
    m["postura"] = m["postura"].replace("desconocida", pd.NA).ffill().fillna("desconocida")
    m.insert(0, "tiempo_s", [round(i * seg_por_frame, 2) for i in range(len(m))])
    return m[["tiempo_s", "postura", "alt_cuello", "alt_anca"]]


def segmentos(linea: pd.DataFrame, seg_por_frame: float):
    """Agrupa frames consecutivos con la misma postura: (postura, inicio, fin)."""
    res, ini = [], 0
    p = linea["postura"].tolist()
    for i in range(1, len(p) + 1):
        if i == len(p) or p[i] != p[ini]:
            res.append((p[ini], ini * seg_por_frame, i * seg_por_frame))
            ini = i
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--fps", type=float, default=2.0, help="frames por segundo a analizar (default 2)")
    ap.add_argument("--salida", type=Path, default=Path("resultados"), help="carpeta de resultados")
    ap.add_argument("--sin-video", action="store_true", help="no generar el video con los puntos dibujados")
    args = ap.parse_args()

    video = args.video.resolve()
    carpeta = (args.salida / video.stem).resolve()
    carpeta.mkdir(parents=True, exist_ok=True)

    reducido = carpeta / f"{video.stem}_reducido.mp4"
    print(f"[1/3] Reduciendo el video a {args.fps} fps...")
    seg_por_frame = submuestrear(video, reducido, args.fps)

    print("[2/3] Detectando keypoints (la primera vez descarga el modelo)...")
    import deeplabcut  # se importa aqui para que --help y los pasos 1 y 3 no esperen a DLC

    dlc_dir = carpeta / "keypoints"
    dlc_dir.mkdir(exist_ok=True)
    if not list(dlc_dir.glob("*.h5")):
        deeplabcut.video_inference_superanimal(
            [str(reducido)],
            superanimal_name="superanimal_quadruped",
            model_name="hrnet_w32",
            detector_name="fasterrcnn_resnet50_fpn_v2",
            video_extensions=".mp4",
            dest_folder=str(dlc_dir),
            pcutoff=0.6,
            batch_size=8,
            detector_batch_size=1,
            max_individuals=1,
            plot_bboxes=False,
            create_labeled_video=not args.sin_video,
        )

    print("[3/3] Clasificando la postura...")
    h5 = sorted(dlc_dir.glob("*.h5"))[0]
    linea = postura_desde_h5(h5, seg_por_frame)
    csv = carpeta / f"{video.stem}_postura.csv"
    linea.to_csv(csv, index=False)

    print(f"\nResultado para {video.name}  ({len(linea)} frames analizados)")
    for post, n in linea["postura"].value_counts().items():
        print(f"  {post:12s} {n / len(linea) * 100:5.1f}% del tiempo")
    print("\nLinea de tiempo:")
    for post, a, b in segmentos(linea, seg_por_frame):
        print(f"  {a:6.1f}s - {b:6.1f}s  {post}")
    print(f"\nCSV: {csv}")


if __name__ == "__main__":
    main()
