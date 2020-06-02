import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

from utils import reverse_normalize, normalize_transform, _is_iterable
from torchvision import transforms


def plot_prediction_grid(model, images, dim=None, figsize=None, score_filter=0.6):
    """Plots a grid of images with boxes drawn around predicted objects.

    :param model: The trained model with which to run object detection.
    :type model: core.Model
    :param images: An iterable of images to plot. If the images are
        normalized torch.Tensor images, they will automatically be
        reverse-normalized and converted to PIL images for plotting.
    :type images: iterable
    :param dim: (Optional) The dimensions of the grid in the format
        ``(rows, cols)``. If no value is given, the grid is of the shape
        ``(len(images), 1)``. ``rows * cols`` must match the number of
        given images, or a ValueError is raised. Defaults to None.
    :type dim: tuple or None
    :param figsize: (Optional) The size of the entire grid in the format
        ``(width, height)``. Defaults to None.
    :type figsize: tuple or None
    :param score_filter: (Optional) Minimum score required to show a
        prediction. Defaults to 0.6.
    :type score_filter: float

    """

    # If not specified, show all in one column
    if dim is None:
        dim = (len(images), 1)

    if dim[0] * dim[1] != len(images):
        raise ValueError('Grid dimensions do not match size of list of images')

    fig, axes = plt.subplots(dim[0], dim[1], figsize=figsize)

    # Loop through each image and position in the grid
    index = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            image = images[index]
            preds = model.predict(image)

            # If already a tensor, reverse normalize it and turn it back
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(reverse_normalize(image))
            index += 1

            # Get the correct axis
            if dim[0] <= 1 and dim[1] <= 1:
                ax = axes
            elif dim[0] <= 1:
                ax = axes[j]
            elif dim[1] <= 1:
                ax = axes[i]
            else:
                ax = axes[i, j]

            ax.imshow(image)

            # Plot boxes and labels
            for label, box, score in zip(*preds):
                if score >= score_filter:
                    width, height = box[2] - box[0], box[3] - box[1]
                    initial_pos = (box[0], box[1])
                    rect = patches.Rectangle(initial_pos, width, height, linewidth=1,
                                             edgecolor='r', facecolor='none')
                    ax.add_patch(rect)

                    ax.text(box[0] + 5, box[1] - 10, '{}: {}'
                            .format(label, round(score.item(), 2)), color='red')
                ax.set_title('Image {}'.format(index))

    plt.show()


def show_labeled_image(image, boxes, labels=None):
    """Show the image along with the specified boxes around detected objects.
    Also displays each box's label if a list of labels is provided.

    :param image: The image to plot. If the image is a normalized
        torch.Tensor object, it will automatically be reverse-normalized
        and converted to a PIL image for plotting.
    :type image: numpy.ndarray or torch.Tensor
    :param boxes: A torch tensor of size (N, 4) where N is the number
        of boxes to plot, or simply size 4 if N is 1.
    :type boxes: torch.Tensor
    :param labels: (Optional) A list of size N giving the labels of
            each box (labels[i] corresponds to boxes[i]). Defaults to None.
    :type labels: torch.Tensor or None

    """

    fig, ax = plt.subplots(1)
    # If the image is already a tensor, convert it back to a PILImage
    # and reverse normalize it
    if isinstance(image, torch.Tensor):
        image = reverse_normalize(image)
        image = transforms.ToPILImage()(image)
    ax.imshow(image)

    # Show a single box or multiple if provided
    if boxes.ndim == 1:
        boxes = boxes.view(1, 4)

    if labels is not None and not _is_iterable(labels):
        labels = [labels]

    # Plot each box
    for i in range(boxes.shape[0]):
        box = boxes[i]
        width, height = (box[2] - box[0]).item(), (box[3] - box[1]).item()
        initial_pos = (box[0].item(), box[1].item())
        rect = patches.Rectangle(initial_pos,  width, height, linewidth=1,
                                 edgecolor='r', facecolor='none')
        if labels:
            ax.text(box[0] + 5, box[1] - 5, '{}'.format(labels[i]), color='red')

        ax.add_patch(rect)

    plt.show()
