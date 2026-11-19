# Prototipo 1 (solo para pruebas)

Primer modelo entrenado del proyecto. **No es el modelo final**: se publico solo para tener algo con que probar el pipeline.

## Que es
- Motor: DeepLabCut 3.0.1 (PyTorch), red `resnet50_gn`, 200 epocas, entrenado en GPU (Kaggle T4).
- Datos: StanfordExtra, solo 3 razas (German_shepherd, Siberian_husky, beagle): 354 imagenes (336 train / 18 test).
- Checkpoint incluido: `snapshot-best-160.pt` (el de mayor `test.mAP`, elegido sobre las mismas 18 imagenes de prueba, asi que la metrica es algo optimista).
- `config.yaml` define 24 bodyparts, pero StanfordExtra solo anota 20: left_eye, right_eye, withers y throat tienen 0 % de anotaciones, por lo que este modelo nunca los aprendio.

## Metricas (checkpoint best-160, 18 imagenes de prueba)
| | train | test |
|---|---|---|
| rmse (px) | 6.92 | 31.35 |
| rmse_pcutoff (px) | 1.61 | 7.75 |
| mAP | 96.35 | 69.95 |
| mAR | 97.47 | 73.89 |

## Limitaciones conocidas (probado en un video de 1080p)
- Detecta con confianza (> 0.6) solo nariz, barbilla, base de orejas y patas delanteras.
- Patas traseras, cola y los 4 puntos sin anotaciones casi nunca superan 0.6.
- En CPU tardo ~9 s por cuadro (analisis offline, no tiempo real).

## Como probarlo
1. En `config.yaml` y en `dlc-models-pytorch/.../train/pytorch_config.yaml` cambia `project_path` (y `pose_config_path`) por la ruta absoluta de esta carpeta en tu equipo.
2. Con DeepLabCut 3.0.1 instalado:

```python
import deeplabcut
cfg = r"RUTA/A/Prototipo 1/config.yaml"
deeplabcut.analyze_videos(cfg, ["mi_video.mp4"], shuffle=1, save_as_csv=True)
deeplabcut.create_labeled_video(cfg, ["mi_video.mp4"], shuffle=1, draw_skeleton=False)
```
