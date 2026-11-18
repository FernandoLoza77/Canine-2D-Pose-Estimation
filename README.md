# Canine-2D-Pose-Estimation

Herramienta para veterinaria que, a partir de un video de un perro, indica en qué
postura está a lo largo del tiempo: **acostado**, **sentado** o **parado**.

Usa el modelo preentrenado **SuperAnimal-Quadruped** de [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)
3.0.1 (39 puntos del cuerpo, motor PyTorch) y reglas geométricas sobre esos puntos.
No necesita entrenar nada.

> **Estado: prueba de concepto.** Validada con solo 4 videos cortos (vista lateral, un
> perro). Ver [Limitaciones](#limitaciones) antes de usarla en un caso real.

## Instalación

Requisitos: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) e internet
(la primera vez se descargan ~300 MB de pesos desde Hugging Face).

```bash
git clone https://github.com/FernandoLoza77/canine-2D-pose-estimation.git
cd canine-2D-pose-estimation

conda create -n pose_canina python=3.12
conda activate pose_canina

# 1) PyTorch (elige la variante para tu equipo en https://pytorch.org/get-started/locally/)
pip install torch torchvision

# 2) DeepLabCut y dependencias
pip install -r requirements.txt
```

## Uso

```bash
python scripts/detectar_postura.py ruta/al/video.mp4
```

Opciones: `--fps 1` (frames por segundo a analizar, por defecto 2), `--salida carpeta`
(por defecto `resultados/`), `--sin-video` (no genera el video con los puntos).

Resultado en `resultados/<video>/`: un CSV con la postura de cada instante
(`<video>_postura.csv`), el video con los puntos dibujados y, en la terminal, un
resumen (ejemplo ilustrativo del formato, no es un resultado real):

```
  acostado      72.7% del tiempo
  sentado       27.3% del tiempo

  0.0s - 8.0s   acostado
  8.0s - 11.0s  sentado
```

**Velocidad:** en una laptop sin GPU el detector de perros tarda ~10 s por frame
analizado; por eso el video se reduce a 2 fps. Un video de 1 minuto son ~120 frames
(unos 20 minutos). Con GPU es mucho más rápido.

## Cómo funciona

1. `scripts/detectar_postura.py` reduce el video a pocos frames por segundo.
2. SuperAnimal-Quadruped detecta los puntos del cuerpo (patas, cuello, cadera...).
3. `scripts/classify_posture.py` calcula, normalizando por el largo del tronco y tomando
   como suelo la pata más baja:
   - **acostado**: cuello bajo (altura < 0.6),
   - **sentado**: cuello alto y cadera baja (altura < 0.6),
   - **parado**: cuello alto y cadera alta.
   Si el modelo no ve suficientes puntos en un frame, ese frame hereda la postura anterior.

## Limitaciones

- Los umbrales se eligieron mirando **los mismos 4 videos** con los que se probó
  (97 % de frames bien: 85 de 88; los 3 restantes quedaron como "desconocida"). Ese
  número **no es una validación independiente**.
- Pensado para **un solo perro** y **vista lateral**; con dos perros o vista frontal /
  desde arriba los resultados pueden ser incorrectos.
- No distingue otras posturas (por ejemplo "echado de lado" frente a "echado boca abajo").

## Historial del proyecto

Antes de esta versión se entrenó un modelo propio con DeepLabCut sobre
[StanfordExtra](https://github.com/benjiebob/StanfordExtra) (12 razas, 20 puntos,
1046 imágenes, 60 épocas en Kaggle): mAP 64 % en su conjunto de prueba, pero sobre un
video real solo detectaba con confianza la cabeza (patas traseras y cola, 0 %), porque
StanfordExtra anota esos puntos en menos de la mitad de sus imágenes. Por eso se cambió
al modelo preentrenado. Los scripts de ese camino siguen en `scripts/` y `data/`.

## Estructura

```
scripts/detectar_postura.py          herramienta principal (video -> postura)
scripts/classify_posture.py          reglas de postura sobre los keypoints
scripts/analyze_video_superanimal.py prueba del modelo preentrenado sobre videos
scripts/analyze_video.py             inferencia con el modelo entrenado de 12 razas
scripts/build_subset_dataset.py      armado del dataset de StanfordExtra
data/loaders/                        lectura de datasets a un formato común
```
