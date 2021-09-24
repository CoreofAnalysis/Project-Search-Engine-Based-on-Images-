
import numpy as np

from skimage import color, draw, img_as_float

class HistogramExtractor(object):
    """ Class defines a 3D Histogram based feature extractor, optionally incorporating regions.
        The size of the output feature vector depends on the number of
        bins used per channel and whether the regions option is used.
        The size can be calculated using the formula:
        size(feature_vector) = bins_ch_1 * bins_ch_2 * bins_ch_3 * num_regions
    """

    def __init__(self, nbins_per_ch=(16,), use_hsv=False, use_regions=False, radius=0.6):
        """
        Initialize Histogram extractor.
        :param nbins_per_ch: number of bins per channel (tuple)
        :param use_hsv: whether to convert image to HSV space (boolean)
        :param use_regions: whether image should be split into regions (boolean)
        :param radius: radius of central part of image (float: 0.0-1.0)
        """
        self._bins_per_ch = nbins_per_ch if len(nbins_per_ch) == 3 else 3 * nbins_per_ch
        self._use_regions = use_regions
        self._use_hsv = use_hsv
        self._radius = radius

    def _preprocess_image(self, image):
        # convert image to floats
        image_flt = img_as_float(image)
        if self._use_hsv:
            image_flt = color.rgb2hsv(image_flt)

        return image_flt

    def _extract_regions(self, image):
        height, width = image.shape[:2]
        corners = [
            (0, height // 2, 0, width // 2),  # top left corner
            (height // 2, height, 0, width // 2),  # bottom left corner
            (height // 2, height, width // 2, width),  # bottom right corner
            (0, height // 2, width // 2, width)  # top right corner
        ]

        # cutout center region
        r, c = draw.ellipse(height // 2, width // 2, int(self._radius * height / 2),
                            int(self._radius * width / 2))
        center_mask = np.zeros(image.shape)
        center_mask[r, c, :] = 1
        regions = [image * center_mask]
        # cutout corner regions
        for (start_x, end_x, start_y, end_y) in corners:
            corner_mask = np.zeros(image.shape)
            corner_mask[start_x:end_x, start_y:end_y, :] = 1
            regions.append(image * corner_mask)
        return regions

    def extract(self, image):
        """
        Extracts abstract features from the given image.
        :param image: image from which features should be extracted
        :return: a numpy array with features, dimensionality depends on class settings.
        """
        # preprocess image (change to float, possibly move to hsv, etc.)
        image = self._preprocess_image(image)

        # split image into regions if necessary
        image_regions = self._extract_regions(image) if self._use_regions else [image]

        regions_hists = []
        for region in image_regions:
            # compute channel histograms for each region
            hists, _ = np.histogramdd(region.reshape(-1, 3), bins=self._bins_per_ch,
                                      range=[(0., 1.)]*3, normed=True)
            regions_hists.extend(hists.reshape(-1))

        return np.array(regions_hists)

