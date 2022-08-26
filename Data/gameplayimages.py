from datasets import load_dataset
import matplotlib.pyplot as plt

gameplay_ds = load_dataset("Bingsu/Gameplay_Images").shuffle()

GAMEPLAY_SHAPE = (640, 360)

'''
split: 'train'
{
    'image': PIL Image,
    'label': int
}
'''

gameplay_labels = (
    "Among Us",
    "Apex Legends",
    "Fortnite",
    "Forza Horizon",
    "Free Fire",
    "Genshin Impact",
    "God of War",
    "Minecraft",
    "Roblox",
    "Terraria"
)

if __name__ == "__main__":
    for i in range(10):
        plt.imshow(gameplay_ds["train"][i]["image"])
        plt.title(gameplay_labels[gameplay_ds["train"][i]["label"]])
        plt.show()
