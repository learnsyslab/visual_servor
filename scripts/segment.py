import time
from ultralytics import YOLO
from ultralytics.models.fastsam import FastSAMPredictor
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from cv_bridge import CvBridge
import ros_numpy

import IPython

MODEL_RGB_IMAGE_WIDTH = 640
MODEL_RGB_IMAGE_HEIGHT = 480
MODEL_RGB_IMAGE_SIZE = (MODEL_RGB_IMAGE_WIDTH, MODEL_RGB_IMAGE_HEIGHT)

class Person:
    def __init__(self, image, contours):
        self.mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(self.mask, [contours], -1, 1, cv2.FILLED)
        self.mask = self.mask.astype(bool)
        self.center = np.mean(np.where(self.mask.T), axis=1).astype(np.int32)


class CameraNode:
    def __init__(self):
        self.model = YOLO("yolo11n-seg.pt")
        self.bridge = CvBridge()

        self.rgb_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.rgb_callback, queue_size=1
        )
        self.points_sub = rospy.Subscriber(
            "/camera/depth_registered/points",
            PointCloud2,
            self.points_callback,
            queue_size=1,
        )

        self.rgb_image = None
        self.person = None

    def points_callback(self, msg):
        if self.person is None:
            return

        # TODO get the actual depth data
        data = ros_numpy.numpify(msg)
        depth = data["z"]
        depth = cv2.resize(depth, MODEL_RGB_IMAGE_SIZE)
        print(np.median(depth[self.person.mask]))

        # # scale from network image size to camera image size
        # scale = (msg.width / MODEL_RGB_IMAGE_WIDTH, msg.height / MODEL_RGB_IMAGE_HEIGHT)
        # uv = (self.target * scale).astype(int).tolist()
        #
        # # get corresponding 3D points
        # # TODO how to do this reliably?
        # target3d = pc2.read_points_list(msg, field_names=["x", "y", "z"], uvs=[uv])[0]
        # self.target3d = np.array([target3d.x, target3d.y, target3d.z])
        # # print(self.target3d)

    def rgb_callback(self, msg):
        rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        rgb_image = cv2.resize(rgb_image, MODEL_RGB_IMAGE_SIZE)

        results = self.model.predict(rgb_image, classes=[0], max_det=1, conf=0.5, verbose=False)

        # nothing to do if there are no detections
        if len(results[0].boxes.cls) == 0:
            return

        contours = np.int32([results[0].masks.xy[0]])
        self.person = Person(rgb_image, contours)

        # IPython.embed()
        # raise ValueError()

        rgb_image[self.person.mask, :] = 0
        cv2.circle(rgb_image, self.person.center, 5, [0, 0, 255], -1)
        self.rgb_image = rgb_image

        # # keypoint detector is really slow if no one is in the scene, so we
        # # first just run detection
        # if len(detections[0].boxes.cls) == 0:
        #     return
        #
        # results = self.model.track(self.rgb_image, conf=DET_CONFIDENCE, verbose=False)
        # boxes = results[0].boxes
        #
        # # if the detector is uncertain the person won't be tracked right away,
        # # so we ignore for now
        # if not boxes.is_track:
        #     return
        #
        # track_ids = boxes.id.int().cpu().tolist()
        # keypoints = results[0].keypoints.data.cpu().numpy()
        #
        # # remove people that are no longer detected
        # # note conversion to list because we cannot delete keys while iterating
        # for id in list(self.people.keys()):
        #     if id not in track_ids:
        #         del self.people[id]
        #
        # # add new people or update existing people
        # for i, id in enumerate(track_ids):
        #     if id not in self.people:
        #         # default center is centered in x and 3/4 up in y
        #         x, y, w, h = boxes.xywh[i].cpu().numpy()
        #         center = np.array([x, y - 0.25 * h])
        #         self.people[id] = Person(id=id, center=center, keypoints=keypoints[i])
        #     else:
        #         self.people[id].update(keypoints=keypoints[i])
        #
        #     self.people[id].box_xyxy = boxes.xyxy[i].cpu().numpy()
        #     # print(boxes.conf[i].cpu().numpy())
        #
        # # check for raised hand
        # # current target still valid: it exists and still has a raised hand
        # if self.target_id is not None and self.target_id in self.people:
        #     if self.people[self.target_id].has_hand_raised():
        #         return
        #
        # # target is invalid: check if a new target exists
        # for id, person in self.people.items():
        #     if person.has_hand_raised():
        #         self.target_id = id
        #         return
        #
        # # otherwise we have no target
        # self.target_id = None


    def annotated_image(self):
        annotator = Annotator(self.rgb_image)

        # (shallow) copy to avoid changing dict size during iteration
        people = self.people.copy()
        for track_id, person in people.items():
            annotator.kpts(person.keypoints)
            annotator.box_label(person.box_xyxy, str(track_id))
        return annotator.result()


def main():
    rospy.init_node("camera_node")
    node = CameraNode()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        if node.rgb_image is not None:
            cv2.imshow("image", node.rgb_image)
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break
        rate.sleep()

if __name__ == "__main__":
    main()

# # Create FastSAMPredictor
# overrides = dict(
#     conf=0.25,
#     task="segment",
#     mode="predict",
#     model="FastSAM-s.pt",
#     save=False,
#     imgsz=640,
# )
# predictor = FastSAMPredictor(overrides=overrides)
#
# # Segment everything
# everything_results = predictor("twopeople.jpg")
#
# # Prompt inference
# text_results = predictor.prompt(everything_results, texts="a person")
