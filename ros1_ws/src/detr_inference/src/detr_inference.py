#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from PIL import Image as PILImage, ImageDraw, ImageFont
import numpy as np
import os
import rospkg
import time
import cv2

class DetrInferenceNode:
    def __init__(self):
        rospy.init_node('detr_inference', anonymous=True)
        
        self.rospack = rospkg.RosPack()
        self.bridge = CvBridge()
        
        # Load parameters
        self.load_parameters()
        
        # Load model
        self.load_model()
        
        # Load class names and colors
        self.class_list = self.load_classes()
        self.class_colors = {class_name: self.colors[i % len(self.colors)] for i, class_name in enumerate(self.class_list)}
        
        # Publishers
        self.init_publishers()
        
        # Timer for processing frames
        rospy.Timer(rospy.Duration(0.2), self.timer_callback)

    def load_parameters(self):
        """Load ROS parameters."""
        self.classes_path = rospy.get_param('~classes_path', os.path.join(self.rospack.get_path("detr_inference"), "classes", "boat_classes.txt"))
        self.hub_id = rospy.get_param('~hub_id', "ARG-NCTU")
        self.repo_id = rospy.get_param('~repo_id', "detr-resnet-50-finetuned-20-epochs-boat-dataset")
        self.confidence_threshold = rospy.get_param('~confidence_threshold', 0.8)
        self.sub_camera_topic = rospy.get_param('~sub_camera_topic', '/camera_middle/color/image_raw/compressed')
        self.pub_detection_image_enabled = rospy.get_param('~pub_detection_image', True)

        # Colors for bounding boxes
        self.colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta",
                       "lime", "pink", "teal", "lavender", "brown", "beige", "maroon", "mint",
                       "olive", "apricot", "navy", "grey", "white", "black"]

    def load_classes(self):
        """Load class names from the specified file."""
        try:
            with open(self.classes_path, "r") as f:
                return [cname.strip() for cname in f.readlines()]
        except FileNotFoundError:
            rospy.logerr(f"Class file {self.classes_path} not found!")
            return []

    def load_model(self):
        """Load the DETR model and processor."""
        self.image_processor = AutoImageProcessor.from_pretrained(f"{self.hub_id}/{self.repo_id}")
        self.model = AutoModelForObjectDetection.from_pretrained(f"{self.hub_id}/{self.repo_id}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def init_publishers(self):
        """Initialize ROS publishers."""
        self.pub_detection_image = rospy.Publisher(rospy.get_param('~pub_camera_topic', '/detection_result_img/camera_stitched/compressed'), CompressedImage, queue_size=1)
        self.pub_scores = rospy.Publisher(rospy.get_param('~pub_scores_topic', '~detection_scores'), Float32MultiArray, queue_size=1)
        self.pub_labels = rospy.Publisher(rospy.get_param('~pub_labels_topic', '~detection_labels'), Int32MultiArray, queue_size=1)
        self.pub_boxes = rospy.Publisher(rospy.get_param('~pub_boxes_topic', '~detection_boxes'), Int32MultiArray, queue_size=1)

    def detect_objects(self, image):
        """Perform object detection on the input image."""
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(outputs, threshold=self.confidence_threshold, target_sizes=target_sizes)[0]
        return results

    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on the image."""
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
            rospy.logwarn("Custom font not found, using default font.")

        for score, label, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
            class_name = self.model.config.id2label[label.item()]
            box_color = self.class_colors.get(class_name, "white")
            x, y, x2, y2 = [round(i, 2) for i in box.tolist()]
            draw.rectangle((x, y, x2, y2), outline=box_color, width=2)
            bbox_area = int((x2 - x) * (y2 - y))
            draw.text((x, y-60), f"class: {class_name}, conf: {score:.2f}, area: {bbox_area}", fill=box_color, font=font)

        return image

    def timer_callback(self, event):
        """ROS timer callback function to process incoming images."""
        try:
            msg = rospy.wait_for_message(self.sub_camera_topic, CompressedImage, timeout=0.2)
        except rospy.ROSException:
            rospy.loginfo('No new image received')
            return

        start_time = time.time()
        msg_header = msg.header

        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.loginfo('CvBridgeError: %s', e)
            return

        rospy.loginfo("Image conversion took: %f seconds", time.time() - start_time)

        pil_image = PILImage.fromarray(cv_image)

        if pil_image:
            start_time = time.time()
            detections = self.detect_objects(pil_image)
            rospy.loginfo("Detection processing took: %f seconds", time.time() - start_time)

            if len(detections["scores"]) > 0:
                scores = detections["scores"].detach().cpu().numpy().tolist()
                labels = detections["labels"].detach().cpu().numpy().tolist()
                boxes = detections["boxes"].detach().cpu().numpy().astype(np.int32).tolist()

                # Publish detections
                self.pub_scores.publish(Float32MultiArray(data=scores))
                self.pub_labels.publish(Int32MultiArray(data=labels))
                self.pub_boxes.publish(Int32MultiArray(data=[item for sublist in boxes for item in sublist]))  # Flattened 2D array

                rospy.loginfo("Published detections: %d objects detected", len(scores))

            if self.pub_detection_image_enabled:
                try:
                    processed_image = self.draw_detections(pil_image, detections)
                    ros_image = self.bridge.cv2_to_compressed_imgmsg(np.array(processed_image), dst_format='jpeg')
                    if msg_header is not None:
                        ros_image.header = msg_header
                        ros_image.header.stamp = rospy.Time.now()
                    self.pub_detection_image.publish(ros_image)
                    rospy.loginfo("Total processing time: %f seconds", time.time() - start_time)
                except CvBridgeError as e:
                    rospy.loginfo('CvBridgeError while converting back: %s', e)

if __name__ == '__main__':
    try:
        node = DetrInferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
