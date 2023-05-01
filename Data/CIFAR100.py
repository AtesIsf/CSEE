from datasets import load_dataset
import matplotlib.pyplot as plt

cifar_ds = load_dataset("cifar100").shuffle()

cifar_labels = (
    "Aquatic Mammals", "Fish", "Flowers", "Food Containers", 
    "Fruit and Vegetables", "Household Electrical Device", 
    "Household Furniture", "Insects", "Large Carnivores", 
    "Large Man-made Outdoor Things", "Large Natural Outdoor Scenes", 
    "Large Omnivores and Herbivores", "Medium-sized Mammals", 
    "Non-insect Invertebrates", "People", "Reptiles", "Small Mammals", 
    "Trees", "Vehicles 1", "Vehicles 2"
)
CIFAR_SHAPE = (32, 32)

# Splits: train, test
''' Format (I will use the fine labels)
    'img': PIL Image
    'fine_label': int 
    'coarse_label': int
'''

if __name__ == "__main__":
    for i in range(10):
        plt.imshow(cifar_ds["train"][i]["img"])
        plt.title(cifar_labels[cifar_ds["train"][i]["coarse_label"]])
        plt.show()