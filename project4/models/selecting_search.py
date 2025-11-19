import cv2

class SelectiveSearchExtractor:
    def __init__(self, mode="fast"):
        """
        mode: 'fast' or 'quality'
        """
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        self.mode = mode

    def get_regions(self, img):
        """
        Input:
            img: numpy array (BGR or RGB)
        Output:
            List of region boxes (x, y, w, h)
        """
        self.ss.setBaseImage(img)

        if self.mode == "fast":
            self.ss.switchToSelectiveSearchFast()
        else:
            self.ss.switchToSelectiveSearchQuality()

        rects = self.ss.process()   # list of (x, y, w, h)
        return rects
