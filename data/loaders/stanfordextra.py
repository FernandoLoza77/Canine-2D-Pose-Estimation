import json

def load_stanfordextra(json_path, bodypart_names):
    """
    json_path: ruta al archivo JSON (ej. StanfordExtra_v12.json)
    bodypart_names: lista de 24 nombres, en el mismo orden que 'joints'
                     (ej. ['left_front_paw', 'left_front_knee', ...])

    Regresa: una lista de diccionarios, uno por imagen:
             [{"image_path": ..., "keypoints": {...}}, ...]
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    resultado = []
    for entrada in data:
        image_path = entrada["img_path"] #saca el nombre/ruta de la imagen de esa entrada del JSON
        keypoints = {}#creamos un dic vacio, se guardaran cada keypoint con su nombre

        for nombre, punto in zip(bodypart_names, entrada["joints"]):
            x,y, visibilidad = punto

            if visibilidad == 0:
                keypoints[nombre] = None
            else:
                keypoints[nombre] = (x,y)

        resultado.append({"image_path": image_path, "keypoints": keypoints})



    return resultado

if __name__ == "__main__":
    bodyparts = [
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

    json_path = r"C:\Users\usuario\Desktop\UADY\SemestreAgostoDiciembre2026\Practicas\DataSetStanford\StanfordExtra_V12\StanfordExtra_v12.json"

    resultado = load_stanfordextra(json_path, bodyparts)
    print(len(resultado))
    print(resultado[0])