import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def convert_to_dlc_format(loaded_data, bodypart_names, scorer, source_images_dir, dest_dir):
    """
    loaded_data: lista de {"image_path": ..., "keypoints": {...}}
                 (es la salida de un loader, ej. load_stanfordextra)
    bodypart_names: lista de 24 nombres, mismo orden que en config.yaml
    scorer: nombre del scorer (debe coincidir EXACTO con el de config.yaml)
    source_images_dir: carpeta donde estan las imagenes originales
                        (image_path es relativo a esta carpeta)
    dest_dir: carpeta labeled-data/<nombre> de destino

    Efecto: copia las imagenes a dest_dir (sin subcarpetas) y guarda ahi
            CollectedData_<scorer>.h5 y CollectedData_<scorer>.csv
    """
    source_images_dir = Path(source_images_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # TODO paso 3: aqui vamos a construir dos listas:
    # - row_index_tuples: una tupla por imagen, para el indice de filas
    # - row_values: una lista de coordenadas por imagen, para los valores

    row_index_tuples = []
    row_values = []

    for entrada in loaded_data:
        image_filename = Path(entrada["image_path"]).name
        source_image = source_images_dir / entrada["image_path"]

        if not source_image.exists():
            print(f"Aviso: no se encontro {source_image}, se omite.")
            continue

        shutil.copy2(source_image, dest_dir / image_filename)

        coords = []
        for nombre in bodypart_names:
            punto = entrada["keypoints"][nombre]
            if punto is None:
                coords.extend([np.nan, np.nan])
            else:
                x, y = punto
                coords.extend([x, y])

        row_values.append(coords)
        row_index_tuples.append(("labeled-data", dest_dir.name, image_filename))

        columns = pd.MultiIndex.from_product(
        [[scorer], bodypart_names, ["x", "y"]],
        names=["scorer", "bodyparts", "coords"],
    )
    index = pd.MultiIndex.from_tuples(row_index_tuples)
    df = pd.DataFrame(row_values, index=index, columns=columns)

    h5_path = dest_dir / f"CollectedData_{scorer}.h5"
    csv_path = dest_dir / f"CollectedData_{scorer}.csv"
    df.to_hdf(h5_path, key="df_with_missing", mode="w")
    df.to_csv(csv_path)

    print(f"Listo: {len(row_values)} imagenes procesadas.")
    print(f"Guardado en: {dest_dir}")

    return df

if __name__ == "__main__":
    bodyparts_prueba = ["nose", "left_eye", "right_eye"]

    datos_falsos = [
        {
            "image_path": "raza_x/perro1.jpg",
            "keypoints": {"nose": (10, 20), "left_eye": (5, 5), "right_eye": None},
        },
        {
            "image_path": "raza_x/perro2.jpg",
            "keypoints": {"nose": (30, 40), "left_eye": None, "right_eye": (50, 60)},
        },
    ]

    df = convert_to_dlc_format(
        loaded_data=datos_falsos,
        bodypart_names=bodyparts_prueba,
        scorer="Fernando Loza",
        source_images_dir=".",
        dest_dir="./prueba_salida",
    )
    print(df)