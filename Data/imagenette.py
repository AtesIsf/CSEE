from datasets import load_dataset
import matplotlib.pyplot as plt
import ssl
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context

imagenette_ds = load_dataset("frgfm/imagenette", "320px").shuffle()
IMAGENETTE_SHAPE = (426, 320)

# dict_keys(['train', 'validation'])
'''
    'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=320x320 at 0x19FA12186D8>,
    'label': 'tench',
'''

imagenette_labels = (
    "Tench",
    "English Springer",
    "Cassette Player",
    "Chainsaw",
    "Church",
    "French Horn",
    "Garbage Truck",
    "Gas Pump",
    "Golf Ball",
    "Parachute"
)
# TODO: NORMALIZE DATA OR CHANGE DATASET!!!
if __name__ == "__main__":
    # print(imagenette_ds.keys())
    for i in range(5):
        plt.imshow(imagenette_ds["train"][i]["image"])
        plt.title(imagenette_labels[imagenette_ds["train"][i]["label"]])
        print(imagenette_ds["train"][i]["image"].size)
        plt.show()