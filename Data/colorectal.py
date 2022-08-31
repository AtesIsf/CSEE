import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

col_ds, col_info = tfds.load("colorectal_histology", split="train", shuffle_files=True, with_info=True)

'''FeaturesDict({
    'filename': Text(shape=(), dtype=tf.string),
    'image': Image(shape=(150, 150, 3), dtype=tf.uint8),
    'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=8),
})'''

COL_SHAPE = (150, 150) # ,3

# You can also use take() and a foreach loop on the sample instead of show_examples()
if __name__ == "__main__":
    tfds.show_examples(col_ds, col_info)