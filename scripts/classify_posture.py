"""
Clasifica la postura (acostado / sentado / parado) a partir de los keypoints
de SuperAnimal-Quadruped (archivo .h5 que genera analyze_video_superanimal.py).

Reglas (vista lateral), con medidas normalizadas por el largo del tronco L
(distancia cuello-anca), y "suelo" = la pata mas baja de la imagen:
  alt_cuello = (suelo - y_cuello)/L     alt_anca = (suelo - y_anca)/L
  acostado: cuello bajo        (alt_cuello < UMBRAL_CUELLO)
  sentado : cuello alto y anca baja (alt_anca < UMBRAL_ANCA)
  parado  : cuello alto y anca alta

AVISO: los umbrales se eligieron viendo solo 4 videos cortos. Hay que validarlos
con mas perros y otros angulos antes de darlos por buenos.

Uso:
    python scripts/classify_posture.py ruta/al/archivo.h5 [otro.h5 ...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

UMBRAL_CUELLO = 0.60
UMBRAL_ANCA = 0.60
MIN_CONF = 0.3

PATAS = ["front_left_paw", "front_right_paw", "back_left_paw", "back_right_paw"]
NECESARIOS = ["neck_base", "back_end"]


def medidas_por_frame(d: pd.DataFrame) -> pd.DataFrame:
    d = d.droplevel([0, 1], axis=1)  # quita scorer e individuo
    filas = []
    for i in range(len(d)):
        def pt(nombre):
            x, y, c = (d[nombre][k].iloc[i] for k in ("x", "y", "likelihood"))
            return np.array([x, y]) if c >= MIN_CONF else None

        cuello, anca = pt("neck_base"), pt("back_end")
        patas_y = [p[1] for p in map(pt, PATAS) if p is not None]
        if cuello is None or anca is None or not patas_y:
            filas.append(dict(alt_cuello=np.nan, alt_anca=np.nan))
            continue
        L = np.linalg.norm(cuello - anca)
        if L < 5:
            filas.append(dict(alt_cuello=np.nan, alt_anca=np.nan))
            continue
        suelo = max(patas_y)
        filas.append(dict(alt_cuello=(suelo - cuello[1]) / L,
                          alt_anca=(suelo - anca[1]) / L))
    return pd.DataFrame(filas)


def clasificar(m: pd.DataFrame) -> pd.Series:
    def regla(r):
        if np.isnan(r.alt_cuello):
            return "desconocida"
        if r.alt_cuello < UMBRAL_CUELLO:
            return "acostado"
        return "sentado" if r.alt_anca < UMBRAL_ANCA else "parado"
    return m.apply(regla, axis=1)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        d = pd.read_hdf(arg)
        etiquetas = clasificar(medidas_por_frame(d))
        print(f"\n{Path(arg).parent.name}: {len(etiquetas)} frames")
        print(etiquetas.value_counts().to_string())
