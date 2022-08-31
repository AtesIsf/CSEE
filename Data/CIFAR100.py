from datasets import load_dataset
import matplotlib.pyplot as plt

cifar_ds = load_dataset("cifar100").shuffle()

cifar_labels = (
    "Apple", "Aquarium Fish", "Baby", "Bear", "Beaver", "Bed", "Bee",
    "Beetle", "Bicycle", "Bottle", "Bowl", "Boy", "Bridge", "Bus",
    "Butterfly", "Camel", "Can", "Castle", "Caterpillar", "Cattle",
    "Chair", "Chimpanzee", "Clock", "Cloud", "Cockroach", "Couch",
    "Cra", "Crocodile", "Cup", "Dinosaur", "Dolphin", "Elephant",
    "Flatfish", "Forest", "Fox", "Girl", "Hamster", "House", "Kangaroo",
    "Keyboard", "Lamp", "Lawn Mower", "Leopard", "Lion", "Lizard",
    "Lobster", "Man", "Maple Tree", "Motorcycle", "Mountain", "Mouse",
    "Mushroom", "Oak Tree", "Orange", "Orchid", "Otter", "Palm_tree", "Pear",
    "Pickup Truck", "Pine_tree", "Plain", "Plate", "Poppy", "Porcupine", "Possum",
    "Rabbit", "Raccoon", "Ray", "Road", "Rocket", "Rose", "Sea", "Seal", "Shark",
    "Shrew", "Skunk", "Skyscraper", "Snail", "Snake", "Spider", "Squirrel", 
    "Streetcar", "Sunflower", "Sweet Pepper", "Table", "Tank", "Telephone",
    "Television", "Tiger", "Tractor", "Train", "Trout", "Tulip", "Turtle", 
    "Tardrobe", "Thale", "Willow Tree", "Wolf", "Woman", "Worm",
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
        plt.title(cifar_labels[cifar_ds["train"][i]["fine_label"]])
        plt.show()