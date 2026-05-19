import numpy as np
import matplotlib.pyplot as plt
import os
from ipywidgets import interact

class Raw:
    def __init__(self, value):
        self._value = value.copy()
        self._value.setflags(write=False)
        
    # Make sure to use copy() when assigning values!
    @property
    def value(self):
        ret = self._value.copy()
        ret.setflags(write=True)
        return ret


class DepthMask:
    def __init__(self):
        self.threshold = None
    
    def __call__(self, images, threshold, tmap=None):
        print("Masking...", end='')
        if tmap is not None:
            threshold = tmap
        else:   
            median = np.median(np.squeeze(images), axis=0)
            threshold = median * threshold
        self.threshold = threshold
        for i in range(len(images)):
            mask = np.squeeze(images[i]) < threshold
            images[i] *= mask
        print("Done")

        if tmap is None:
            plt.imshow(threshold)
            plt.title("Threshold map")
            plt.axis('off')
            plt.show()
        return images


class Crop:
    def __init__(self):
        self.bbx = None
        self.ctr = None
        self.cimg = None
    
    def __call__(self, images, min_area=0, cvt_bbx=True, show=False):
        """
        Calculate cropped images, bounding boxes, center coordinates, depth.\n
        :param min_area: 0
        :param show: whether show images with bounding boxes
        :return: bbx, center, depth, c_img
        """
        self.bbx = np.zeros((len(images), 4), dtype=float)
        self.ctr = np.zeros((len(images), 3), dtype=float)
        self.cimg = np.zeros((len(images), 1, 128, 128), dtype=np.uint8)
        
        # Handle different data types
        if images.dtype == np.float32:
            img_normalized = (images * 255).astype(np.uint8)
        elif images.dtype == np.uint16:
            img_normalized = (images / 256).astype(np.uint8)
        else:
            raise ValueError(f"Only float32 and uint16 are supported, got {img.dtype}")

        h_, w_ = img_normalized.shape[1], img_normalized.shape[2]

        for i in tqdm(range(len(images))):
            img = np.squeeze(img_normalized[i])

            # Threshold the normalized image
            (T, timg) = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(timg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 0:
                contour = max(contours, key=lambda x: cv2.contourArea(x))
                area = cv2.contourArea(contour)

                if area < min_area:
                    # print(area)
                    pass

                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    w = 128 if w > 128 else w
                    
                    cropped = img[y:y + h, x:x + w]

                    self.bbx[i] = x / 226, y / 128, w / 226, h / 128
                    if cvt_bbx:
                        self.bbx[i, 2] += self.bbx[i, 0]
                        self.bbx[i, 3] += self.bbx[i, 1]
                        
                    # (x, y, d)
                    self.ctr[i, 0] = (x + w / 2) / 226
                    self.ctr[i, 1] = (y + h / 2) / 128
                    
                    image = np.pad(cropped, pad_width=((64 - h // 2,
                                                        64 - h + h //2), 
                                                       (64 - w // 2,
                                                       64 - w + w // 2)),
                                   mode='constant',
                                   constant_values = 0
                                   )

                    self.cimg[i, 0] = image

                    if show:
                        img = cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
                                            (x, y),
                                            (x + w, y + h),
                                            (0, 255, 0), 1)
                        cv2.namedWindow('Raw Image', cv2.WINDOW_AUTOSIZE)
                        cv2.imshow('Raw Image', img)
                        key = cv2.waitKey(33) & 0xFF
                        if key == ord('q'):
                            break
        print("Done")

    def save(self, save_path):
        
        for item in ('cimg', 'bbx', 'ctr'):
            if getattr(self, item) is not None:
                print(f'Saving {item}...', end='')
                np.save(os.path.join(save_path), getattr(self, item))
                print('Done')


class ImageLoader:
    # Camera time should have been calibrated

    def __init__(self, img_path=None, img=None, *args, **kwargs):
       
        self.img_path = img_path
        self.name = 'Image'
        if img_path is not None:
            self.load_images()
            self.name = os.path.basename(self.img_path)
        else:
            self.rawimg = img

        self.raw_images = Raw(self.rawimg)
        
        self.threshold_map = None
        self.depthmask = DepthMask()

    def load_images(self):
        self.rawimg = np.load(self.img_path)
        print(f" Loaded {self.name} of {self.rawimg.shape} as {self.rawimg.dtype}")
    
    def reset_data(self):
        self.rawimg = self.raw_images.value
        self.threshold_map = None
        print('Data reset!')
    
    def playback(self, compare=False):
        for i, image in enumerate(self.rawimg):
            plt.clf()
            if compare:
                image = np.hstack((image, self.raw[i]))
                plt.title(f"<Masked, Raw> {i} of {len(self.rawimg)}")
            else:
                plt.title(f"Image {i} of {len(self.rawimg)}")
            plt.axis('off')
            plt.imshow(image)
            plt.show()
                
            if self.jupyter_mode:
                clear_output(wait=True)
                # display(plt.gcf())'
            else:
                plt.pause(0.1)

    def depth_mask(self, threshold=0.5, tmap=None):
        self.rawimg = self.depthmask(self.rawimg, threshold, tmap)
        print(f'Max depth = {np.max(self.rawimg)}, Min depth = {np.min(self.rawimg)}')
        return self.depthmask.threshold
        
    def threshold_depth(self, threshold=3000):
        print(f'Thresholding IMG within {threshold}...', end='')
        self.rawimg = np.clip(self.rawimg, 0, threshold)
        print('Done')
        
    def clear_excessive_depth(self, threshold=3000):
        print(f'Clearing depth > {threshold}...', end='')
        self.rawimg[self.rawimg > threshold] = 0
        print('Done')
        
    def normalize_depth(self, threshold=3000):
        print(f'Normalizing depth by {threshold}...', end='')
        self.rawimg = (self.rawimg * (65535 / threshold)).astype(np.uint16)
        print('Done')
        
    def binarize(self):
        self.rawimg[self.rawimg > 0] = 65535
        self.rawimg = self.rawimg.astype(np.float32)
        print('Binarization done')
        
    def save_images(self, save_path, replace='depthimg'):
        
        if self.rawimg is not None:
            print(f'Saving {self.name} dimg...', end='')
            np.save(os.path.join(save_path, self.name.replace(replace, 'dimg')), self.rawimg)
            print('Done')
        
    def browse_images(self, bound=None):
        
        if bound is not None:
            start, end = bound
            n = end - start
        else:
            n = len(self.rawimg)
            start, end = 0, n
            
        if np.array_equal(self.raw_images.value[start:end], self.rawimg[start:end]):
            im = self.rawimg[start:end]
        else:
            im = np.hstack((self.raw_images.value[start:end],
                            self.rawimg[start:end]))
        
        def view_image(i):
            # plt.imshow(self.rawimg[start:end][i], cmap=plt.get_cmap('Blues'))

            plt.imshow(im[i], cmap=plt.get_cmap('Blues'))
            plt.title(f"Image {i} of {n}")
            plt.axis('off')
            plt.show()
        interact(view_image, i=(0, n-1))