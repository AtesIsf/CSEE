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

imagenette_train = []
imagenette_test = []

for i in tqdm(range(len(imagenette_ds["train"][:]["image"]))):
    img = imagenette_ds["train"][i]["image"]
    if img.size[0] == 320: 
        img = img.rotate(90, expand=1)
    img = img.crop((0, 0, 426, 320))
    imagenette_train.append(img)

for i in tqdm(range(len(imagenette_ds["validation"][:]["image"]))):
    img = imagenette_ds["validation"][i]["image"]
    if img.size[0] == 320: 
        img = img.rotate(90, expand=1)
    img = img.crop((0, 0, 426, 320))
    imagenette_test.append(img)


if __name__ == "__main__":
    # print(imagenette_ds.keys())
    for i in range(10):
        # plt.imshow(imagenette_ds["train"][i]["image"])
        plt.imshow(imagenette_train[i])
        plt.title(imagenette_labels[imagenette_ds["train"][i]["label"]])
        print(f"Actual Shape of Image {i+1}: {imagenette_ds['train'][i]['image'].size}")
        print(f"Normalized Shape of Image {i+1}: {imagenette_train[i].size}")
        plt.show()