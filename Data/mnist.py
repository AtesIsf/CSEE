from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

ds = load_dataset("mnist").shuffle()
SHAPE = (28, 28, 1)
# _beans_builder = load_dataset_builder("beans")
# dict_keys(['train', 'validation', 'test'])

''' DATA KEYS
  'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x27601169DD8>,
  'label': 9
'''

N_CLASSES = 10

if __name__ == "__main__":
    # print(_beans_builder.info.features)
    # print(ds.keys())
    for i in range(10):
        plt.imshow(ds["train"][i]["image"])
        plt.show()