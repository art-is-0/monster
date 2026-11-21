from pathlib import Path
from utils import Region


SEED = 67

ROOT = Path(__file__).resolve().parent  # Repo root

FIGURES_PATH = ROOT / "figures"
RESULTS_PATH = ROOT / "results"
PICTURE_PATH = ROOT / "pictures"

FIGURES_PATH.mkdir(exist_ok=True)
RESULTS_PATH.mkdir(exist_ok=True)


PERM_DATA = {
    'train_path' : PICTURE_PATH / 'Bilde2.png',
    'test_path' : PICTURE_PATH / 'Bilde3.png',
    'regions' : [
            Region(2100, 500, 2800, 1200, 0),  # Region for red folder
            Region(700, 900, 1400, 1600, 1),  # Region for blue folder
            Region(0, 0, 900, 700, 2)     # Region for background
        ],
    'name' : 'perm',
    'figx' : 3, 'figy' : 4.5
}

HALL_DATA = {
    'train_path' : PICTURE_PATH / 'hall_train.jpg',
    'test_path' : PICTURE_PATH / 'hall_test.jpg',
    'regions' : [
            Region(300, 2000, 1800, 3500, 0),  # Region for ground
            Region(0, 0, 2050, 500, 1),  # Region for wall
            Region(1635, 835, 2035, 875, 2)     # Region for bars
        ],
    'name' : 'hall',
    'figx' : 3, 'figy' : 4.5
}

WALL_DATA = {
    'train_path' : PICTURE_PATH / 'wall_train.jpg',
    'test_path' : PICTURE_PATH / 'wall_test.jpg',
    'regions' : [
            Region(1050, 650, 1383, 983, 0),  # Region for grass
            # Region(870, 525, 920, 725, 1),  # Region for fence
            Region(330, 730, 1330, 985, 1),  # Region for fence
            Region(0, 400, 455, 750, 2),     # Region for stone
            Region(520, 0, 970, 450, 3)     # Region for sky
        ]
}