from datasets import load_dataset
import matplotlib.pyplot as plt

ds = load_dataset("cifar10").shuffle()

N_CLASSES = 10

SHAPE = (32, 32, 3)

# Splits: train, test
''' Format (I will use the fine labels)
    'img': PIL Image
    'label': int
'''

if __name__ == "__main__":
    for i in range(10):
        plt.imshow(ds["train"][i]["img"])
        plt.title(ds["train"][i]["label"])
        plt.show()