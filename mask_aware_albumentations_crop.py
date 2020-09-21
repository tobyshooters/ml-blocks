import albumentations as albu
import albumentations.augmentations.functional as F

class MaskAwareRandomResizedCrop(albu.DualTransform):
    def __init__(self, min_max_height, height, width, always_apply=False, p=1.0):
        super(MaskAwareRandomResizedCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.min_max_height = min_max_height

    def apply(self, img, min_x, max_x, min_y, max_y, **params):
        crop = img[min_y:max_y, min_x:max_x]
        resized_crop = F.resize(crop, self.height, self.width)
        return resized_crop

    def get_params_dependent_on_targets(self, params):
        """ Image and mask are (H, W, 3), and (H, W) np.array's in (0, 255) """

        # Get bbox around mask
        xs = np.any(params["mask"], axis=0)
        ys = np.any(params["mask"], axis=1)
        mask_min_x, mask_max_x = np.where(xs)[0][[0, -1]]
        mask_min_y, mask_max_y = np.where(ys)[0][[0, -1]]

        # Choose center of crop, within bbox
        cx = np.random.randint(mask_min_x, mask_max_x + 1)
        cy = np.random.randint(mask_min_y, mask_max_y + 1)

        # Crop size from min_max_height
        crop_size =  np.random.randint(self.min_max_height[0], self.min_max_height[1])

        # Get a crop region
        h, w = params["mask"].shape[:2]
        dd = crop_size // 2
        min_x = max(cx - dd, 0)
        max_x = min(cx + dd, w)
        min_y = max(cy - dd, 0)
        max_y = min(cy + dd, h)

        params = { "min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y }
        return params

    @property
    def targets_as_params(self):
        return ["mask"]
