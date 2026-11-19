import deeplabcut

CONFIG_PATH = r"C:\Users\usuario\Desktop\UADY\SemestreAgostoDiciembre2026\Practicas\canine-2D-pose-estimation\dlc_project\config.yaml"

if __name__ == "__main__":
    deeplabcut.create_training_dataset(CONFIG_PATH, num_shuffles=1)
