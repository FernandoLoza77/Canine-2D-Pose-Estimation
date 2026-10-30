import deeplabcut

deeplabcut.create_new_project(
    "Dog_pose_detection",
    "Fernando Loza",
    [r"C:\Users\usuario\Desktop\UADY\SemestreAgostoDiciembre2026\Practicas\perroPaseando.mp4"],
    working_directory=r"C:\Users\usuario\Desktop\UADY\SemestreAgostoDiciembre2026\Practicas\Canine-2D-Pose-Estimation",
    copy_videos=True,
)