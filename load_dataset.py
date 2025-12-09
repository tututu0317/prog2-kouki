from torchvision import datasets
import matplotlib.pyplot as plt

ds_train = datasets.FashionMNIST(
    root='dataset',
    train=True,
    download=True
)

print(f'dataset size: {len(ds_train)}')

image, target = ds_train[0]

print(type(image))
print(target)

plt.imshow(image, cmap='gray_r', vmin=0, vmax=255)
plt.title(target)
plt.show()