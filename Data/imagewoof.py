from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

woof_ds = load_dataset("frgfm/imagewoof", "320px").shuffle()
WOOF_SHAPE = (426, 320)

# Splits = train, validation
'''
  'image': PIL Image
  'label': int
'''

woof_train = []
woof_test = []

for i in tqdm(range(len(woof_ds["train"][:]["image"]))):
    img = woof_ds["train"][i]["image"]
    if img.size[0] == 320: 
        img = img.rotate(90, expand=1)
    img = img.crop((0, 0, 426, 320))
    woof_train.append(img)

for i in tqdm(range(len(woof_ds["validation"][:]["image"]))):
    img = woof_ds["validation"][i]["image"]
    if img.size[0] == 320: 
        img = img.rotate(90, expand=1)
    img = img.crop((0, 0, 426, 320))
    woof_test.append(img)

if __name__ == "__main__":
    for i in range(10):
        # plt.imshow(woof_ds["train"][i]["image"])
        plt.imshow(woof_train[i])
        plt.title(woof_ds["train"][i]["label"])
        print(f"Actual Shape of Image {i+1}: {woof_ds['train'][i]['image'].size}")
        print(f"Normalized Shape of Image {i+1}: {woof_train[i].size}")
        plt.show()