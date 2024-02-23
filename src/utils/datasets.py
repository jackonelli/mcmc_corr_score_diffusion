from src.data.cifar import CIFAR_100_NUM_CLASSES, CIFAR_10_NUM_CLASSES, CIFAR_IMAGE_SIZE, CIFAR_NUM_CHANNELS

def dataset_image_description(dataset):
    if dataset == 'cifar10':
        num_classes, num_channels, image_size = (CIFAR_10_NUM_CLASSES,
                                                 CIFAR_NUM_CHANNELS,
                                                 CIFAR_IMAGE_SIZE)
    elif dataset == 'cifar100':
        num_classes, num_channels, image_size = (CIFAR_100_NUM_CLASSES,
                                                 CIFAR_NUM_CHANNELS,
                                                 CIFAR_IMAGE_SIZE)
    else:
        raise ValueError('Not a valid dataset')
    return num_classes, num_channels, image_size