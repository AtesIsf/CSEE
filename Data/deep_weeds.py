import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

deep_weeds_ds, dw_info = tfds.load("deep_weeds", split="train", shuffle_files=True, with_info=True)

'''FeaturesDict({
    'image': Image(shape=(256, 256, 3), dtype=tf.uint8),
    'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=9),
})'''

DEEP_WEEDS_SHAPE = (256, 256) # ,3

# You can also use take() and a foreach loop on the sample instead of show_examples()
if __name__ == "__main__":
    tfds.show_examples(deep_weeds_ds, dw_info)