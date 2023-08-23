from datasets import load_dataset
import matplotlib.pyplot as plt

ds = load_dataset("keremberke/shoe-classification", name="full").shuffle()

SHAPE = (240, 240, 3)
N_CLASSES = 3

'''
DatasetDict
    train: Dataset
        features: ['image_file_path', 'image', 'labels']
    validation: Dataset
        features: ['image_file_path', 'image', 'labels']
    test: Dataset
        features: ['image_file_path', 'image', 'labels']
'''

if __name__ == "__main__":
    # print(_beans_builder.info.features)
    # print(ds.keys())
    for i in range(5):
        plt.imshow(ds["train"][i]["image"])
        plt.title(ds["train"][i]["labels"])
        plt.show()
    for i in range(5):
        plt.imshow(ds["test"][i]["image"])
        plt.title(ds["test"][i]["labels"])
        plt.show()