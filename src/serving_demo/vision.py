import numpy as np
import cv2

# image size used by the model
MODEL_RGB_IMAGE_WIDTH = 640
MODEL_RGB_IMAGE_HEIGHT = 480
MODEL_RGB_IMAGE_SIZE = (MODEL_RGB_IMAGE_WIDTH, MODEL_RGB_IMAGE_HEIGHT)

# minimum valid depth reading of the camera
MINIMUM_DEPTH = 0.25


class Person:
    def __init__(self, hand_up=False, center=None, depth=0, depth_valid=False):
        self.hand_up = hand_up

        if center is None:
            center = np.zeros(2, dtype=np.int32)
        self.center = np.asarray(center)

        self.depth = depth
        self.depth_valid = depth_valid

    @classmethod
    def from_contours(cls, class_label, contours):
        hand_up = class_label == 0

        # flip because width = columns and height = rows
        self.mask = np.zeros(np.flip(MODEL_RGB_IMAGE_SIZE), dtype=np.uint8)
        cv2.drawContours(self.mask, [contours], -1, 1, cv2.FILLED)
        self.mask = self.mask.astype(bool)

        xs, ys = np.where(self.mask.T)
        x = np.median(xs)

        # we choose a lower quantile for y because we want to aim closer to the
        # head (for servoing in the z-direction)
        y = np.quantile(ys, 0.25)
        center = np.array([x, y], dtype=np.int32)

        return cls(hand_up=hand_up, center=center)

    def update_depth(self, pc_depth):
        depth = cv2.resize(pc_depth, MODEL_RGB_IMAGE_SIZE)
        depth = depth[self.target.mask]
        depth = depth[depth >= MINIMUM_DEPTH]
        if depth.size > 0:
            self.depth_valid = True
            self.depth = np.median(depth)
            print(f"depth = {self.depth}")

    def active(self):
        return self.hand_up and self.depth_valid
