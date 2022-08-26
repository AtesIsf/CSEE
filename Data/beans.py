from datasets import load_dataset
import matplotlib.pyplot as plt

beans_ds = load_dataset("beans").shuffle()
BEANS_SHAPE = (500, 500)
# _beans_builder = load_dataset_builder("beans")
# dict_keys(['train', 'validation', 'test'])

''' DATA KEYS
    'image_file_path': '/root/.cache/huggingface/datasets/downloads/extracted/0aaa78294d4bf5114f58547e48d91b7826649919505379a167decb629aa92b0a/train/bean_rust/bean_rust_train.109.jpg',
    'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=500x500 at 0x16BAA72A4A8>,
    'labels': 1
'''

beans_labels = (
    "Angular Leaf Spot",
    "Bean Rust",
    "Healthy"
)

if __name__ == "__main__":
    # print(_beans_builder.info.features)
    # print(beans_ds.keys())
    for i in range(10):
        plt.imshow(beans_ds["train"][i]["image"])
        plt.title(beans_labels[beans_ds["train"][i]["labels"]])
        plt.show()