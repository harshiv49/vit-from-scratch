# CIFAR-10 data lab

This is a Google Colab-first **data-only** notebook. It downloads CIFAR-10 and gives us a clean place to inspect the images, labels, class balance, and train/test split before adding any modeling code from your own Colab material.

- Notebook: `cifar10_vnn_learning.ipynb`
- Dataset: CIFAR-10
- Training images: 50,000
- Test images: 10,000
- Image shape: 32 × 32 RGB
- Classes: 10

No neural-network architecture, optimizer, loss function, training loop, or evaluation model is included here. Add those later from the Colab files you want to preserve in your second brain.

## Open it in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Choose **File → Upload notebook**.
3. Select `cifar10_vnn_learning.ipynb` from this folder.
4. Run the cells from the top, one at a time.

The dataset is not stored in this repository. Colab downloads it into `/content/cifar10` when the download cell runs.
