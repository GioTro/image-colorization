import numpy as np
import torch
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import color
from torch.functional import F
from typing import Any, NoReturn, Tuple

# dirname = os.path.dirname(__file__)
# pts_in_hull = np.load(os.path.join(dirname, "../misc/npy/pts_in_hull.npy"))
# prior_probs = np.load(os.path.join(dirname, "../misc/npy/prior_probs.npy"))

class Utils:
    @staticmethod
    def load_im(im_path: Path or str) -> np.array:
        """
        Takes path to source and loads rgb image (add dim if greyscale)

        :param im_path: Path to source

        :returns: Image as rgb
        """
        out = np.asarray(Image.open(im_path))
        if out.ndim == 2:
            # Add dim to greyscales
            out = np.tile(out[:, :, None], 3)
        return out

    @staticmethod
    def save_im(save_path: Path or str, im_source: np.array) -> NoReturn:
        """
        Saves im_source at save_path as .png

        :param save_path: Path to destination
        :param im_source: rgb image as np.array
        """
        plt.imsave(str(save_path), im_source)

    @staticmethod
    def preprocess_im(im_rgb: np.array) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Takes the rgb_image, converts to Lab-colorspace and returns the channels as tensors.

        :param im_rgb: rgb image as np.array

        :returns: A tuple with tensors (L_channel, ab_channel)
        """
        im_lab = color.rgb2lab(im_rgb)
        im_L = im_lab[:, :, 0]
        img_ab = im_lab[:, :, 1:]
        X = torch.Tensor(im_L)[None, :, :]
        Y = torch.tensor(np.moveaxis(img_ab, 2, 0), dtype=torch.float32)
        return (X, Y)

    @staticmethod
    def postprocess_tens(L: torch.Tensor, ab: torch.Tensor, mode="bilinear") -> Any:
        """
        Takes L, ab channels as tensors and returns the rgb image

        :param L: L-channel as tensor
        :param ab: ab-channel as tensor

        :returns: rgb image
        """
        out_lab = torch.cat((L, ab), dim=1)
        return color.lab2rgb(out_lab.data.cpu().numpy()[0, ...].transpose((1, 2, 0)))

    # def print_photos(self, epoch):
    #     # testing function
    #     for i in range(3):
    #         name = f"/model/src/test_images/im{i}.jpg"
    #         im_test = util.load_img(name)
    #         X_test, _ = util.preprocess_img(im_test)
    #         X_test = X_test[None, :, :, :]
    #         X_test = X_test.to(device)
    #         y_pred = self.forward(X_test)
    #         im_test = util.postprocess_tens(X_test, y_pred)
    #         util.save_im(f"/model/src/test_images/out/im{i}_{epoch}.jpg", im_test)
