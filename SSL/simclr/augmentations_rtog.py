from PIL import Image
import numpy as np
from torchvision import transforms
import os

class TranslateRotateRTOG(object):

    def __init__(self, translation, rotation, size, window=1, allow_missing=True):

        if isinstance(size, int):
            size = (size, size)
        size = tuple(size)
        if len(size) == 2:
            size += (3,)
        assert translation < min(size[0], size[1])
        self.window = window
        self.size = size
        if len(self.size) == 2:
            self.size += (3,)
        self.translation = translation
        self.allow_missing = allow_missing
        if rotation:
            self.rotation = transforms.RandomRotation([-rotation, rotation], Image.BILINEAR)
        else:
            self.rotation = False

    def _return_location(self, x):
        # w, h = x.split('_')[-3], x.split('_')[-2]
        h, w = x.split('_')[-3], x.split('_')[-2]
        return int(w), int(h)

    def _return_neighbors(self, x, window=1):

        neighbors = []
        w, h = self._return_location(x)
        for d_w in range(-window, window+1):
            for d_h in range(-window, window+1):
                new_neighbor = x.replace(
                        str(w), str(w+d_w)).replace(
                        str(h), str(h+d_h))
#               neighbor_cls = x.split('/')[-2]
#               exists = False
#               for cls in range(6):
#                   alt = new_neighbor.replace('/{}/'.format(neighbor_cls), '/{}/'.format(cls))
#                   if os.path.exists(alt):
#                       new_neighbor, exists = alt, True
#                       break
                if os.path.exists(new_neighbor):
                    neighbors.append(new_neighbor)
                else:
                    neighbors.append('missing')
        return neighbors

    def load_image(self, name, size):

        if name == 'missing':
            return (255 * np.ones(size, dtype=np.uint8))
        else:
            return np.array(Image.open(name).resize(size[:-1]))


    def combine_neighbors(self, x, size=(224, 224, 3), window=1):

        output = []
        neighbors = self._return_neighbors(x, window)
        if not neighbors:
            return None
        neighbors_images = [self.load_image(n, size) for n in neighbors]
        width = 2 * window + 1
        for i in range(width):
            output.append(np.concatenate(neighbors_images[i * width: (i+1) * width], axis=1))
        return Image.fromarray(np.concatenate(output, axis=0))

    def __call__(self, image_name):

        combined = self.combine_neighbors(image_name, self.size, self.window)
        if self.rotation:
            combined = self.rotation(combined)
        t_w, t_h = np.random.choice(np.arange(-self.translation, self.translation), 2)
        tile = np.array(combined)[self.size[0]+t_w: 2 * self.size[0] + t_w,
                       self.size[1] + t_h: 2 * self.size[1] + t_h]
        return Image.fromarray(tile)
