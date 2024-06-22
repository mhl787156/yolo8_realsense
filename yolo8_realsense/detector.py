import numpy as np
import cv2
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
from ultralytics import YOLO  # Assuming YOLOv8 from Ultralytics

import rclpy
from rclpy.node import Node
from rclpy import qos
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo
from realsense2_camera_msgs.msg import RGBD
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

class YoloRealsense(Node):

    def __init__(self):
        super().__init__("yolo_realsense")
        self.declare_parameter("model_path", "model/best.pt")
        self.declare_parameter("detection_rate", 1) #hz
        self.declare_parameter("flower_depth", 20) #mm
        self.declare_parameter("separate_depth_stream", False)
        self.declare_parameter("imshow", False)
        self.declare_parameter("publish_visualisation_markers", False)

        self.declare_parameter("rgb_topic", 'camera/image_raw')
        self.declare_parameter("depth_camera_info_topic", "camera/camera_info")
        self.declare_parameter("depth_topic", 'camera/depth')
        self.declare_parameter("rgbd_topic", 'camera/rgbd')


        self.separate_depth_stream = self.get_parameter("separate_depth_stream").value
        self.flower_depth = self.get_parameter("flower_depth").value

        self.processing_image = False
        self.depth_camera_info = False
        self.last_depth_msg = None

        # Load YOLOv8 model
        model_name = self.get_parameter("model_path").value
        self.model = YOLO(model_name)  # Ensure you have the correct model path and model file

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

        if self.separate_depth_stream:
            self.subscription_rgb = self.create_subscription(Image, self.get_parameter('rgb_topic').value, self.rgbd_image_callback, qos.qos_profile_sensor_data)
            self.subscription_depth = self.create_subscription(Image, self.get_parameter('depth_topic').value, self.separate_depth_callback, qos.qos_profile_sensor_data)
        else:
            self.subscription = self.create_subscription(RGBD, self.get_parameter('rgbd_topic').value, self.rgbd_image_callback, qos.qos_profile_sensor_data)
        
        self.subscription_camera_info = self.create_subscription(CameraInfo, self.get_parameter("depth_camera_info_topic").value, self.camera_info_callback, qos.qos_profile_sensor_data)
        self.publisher = self.create_publisher(Image, 'realsense_detection', 10)
        self.detection_pub = self.create_publisher(Detection3DArray, 'bounding_boxes', 10)

        if self.get_parameter("publish_visualisation_markers").value:
            self.vis_marker_pub = self.create_publisher(MarkerArray, "detected_bbox_markers", 10)
        
        # Have a 1s timer which checks whether a subscriber exists to detection_pub and only starts the subscription if there is one
        self.detection_time = self.get_clock().now()
        self.get_logger().info("Yolo Realsense Node Initialised")

    def bounding_box_to_3d(self, bbox_center, corners_2d, depth_map):
        # Corners 2d is a list of [x, y] coordinates of bbox corners (assumed length 4)
        # Note the bbox center x/y is flipped
        fx = self.depth_camera_info.k[0]
        fy = self.depth_camera_info.k[4]
        cx = self.depth_camera_info.k[2]
        cy = self.depth_camera_info.k[5]

        self.get_logger().info(f"Camera Info: {fx}, {fy}, {cx}, {cy}")

        depth = depth_map[bbox_center[1], bbox_center[0]]

        center_3d = [
            (bbox_center[1] - cx) * depth_map[bbox_center[1], bbox_center[0]] / fx,
            (bbox_center[0] - cx) * depth_map[bbox_center[1], bbox_center[0]] / fy,
            depth
        ]

        self.get_logger().info(f"Center: {center_3d}")

        # Calculate 3D coordinates for each corner
        corners_3d = np.zeros((8, 3))
        for i in range(8):
            # Assume corners 2d is of length 4 with other 4 corners a fixed distance between
            # Not sure abou the 1 in depth_map[corners_2d[i%4, 0], 1]
            corners_3d[i, 0] = (corners_2d[i%4, 1] - cx) * depth_map[corners_2d[i%4, 1], corners_2d[i%4, 0]] / fx
            corners_3d[i, 1] = (corners_2d[i%4, 0] - cy) * depth_map[corners_2d[i%4, 1], corners_2d[i%4, 0]] / fy
            corners_3d[i, 2] = depth + (0 if i < 4 else self.flower_depth)
        
        self.get_logger().info(f"corners_3d: {corners_3d}")

        size_3d = [
            np.linalg.norm(corners_3d[0, :] - corners_3d[1, :]),
            np.linalg.norm(corners_3d[2, :] - corners_3d[3, :]),
            np.linalg.norm(corners_3d[4, :] - corners_3d[5, :])
        ]

        self.get_logger().info(f"Size: {size_3d}")

        return center_3d, corners_3d, size_3d

    def camera_info_callback(self, msg):
        self.depth_camera_info = msg
    
    def separate_depth_callback(self, msg):
        self.last_depth_msg = msg

    def rgbd_image_callback(self, rgbd_msg):

        # self.get_logger().info("Receiving rgb_image")

        timelimit = (1.0/self.get_parameter("detection_rate").value)
        timelim_sec = int(timelimit)
        timelim_ns = (timelimit - timelim_sec) * 1e9
        if self.get_clock().now() - self.detection_time <  Duration(seconds=timelim_sec, nanoseconds=timelim_ns):
            return
        
        # if self.processing_image:
        #     return 
        # self.processing_image = True
        
        self.detection_time = self.get_clock().now()
        
        # frames = pipeline.wait_for_frames()
        if self.separate_depth_stream:
            if not self.last_depth_msg:
                return
            depth_frame = self.br.imgmsg_to_cv2(self.last_depth_msg) #frames.get_depth_frame()
            color_frame = self.br.imgmsg_to_cv2(rgbd_msg) #frames.get_color_frame()
        else:    
            depth_frame = self.br.imgmsg_to_cv2(rgbd_msg.depth) #frames.get_depth_frame()
            color_frame = self.br.imgmsg_to_cv2(rgbd_msg.rgb) #frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame)
        color_image = np.asanyarray(color_frame)

        # Apply colormap on depth image (optional)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        shape = color_image.shape
        depth_colormap_rescaled = cv2.resize(depth_colormap, (shape[1], shape[0]), interpolation= cv2.INTER_NEAREST)

        self.get_logger().info(f"Depth Colourmap: {depth_colormap}")

        # Perform inference on captured color image
        results = self.model(color_image)
        
        # Process results
        detected_boxes = []
        for result in results:
            boxes = result.boxes  # Access the bounding boxes
            for i, box in enumerate(boxes):

                # x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf.cpu().numpy()[0]
                cls = box.cls.cpu().numpy()[0]

                # self.get_logger().info(f"box {i} - cls {cls}, {conf}, [{x1},{y1},{x2},{y2}]")

                if conf > 0.5:  # Filter out low-confidence detections

                    # Get the depth data for the center of the bounding box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    depth = depth_colormap_rescaled[center_y, center_x, 0]  # Note the corrected indexing order
                    # self.get_logger().info(f"Depth is {depth}, {depth_colormap_rescaled.shape}")

                    bbox_center = np.array([center_x, center_y])
                    corners_2d = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
                    center_3d, bbox_3d, size_3d = self.bounding_box_to_3d(bbox_center, corners_2d, depth_colormap_rescaled[:, :, 0])

                    # self.get_logger().info(f"{center_3d}, {bbox_3d}, {size_3d}")
                    self.get_logger().info(f"Flower Detected At {center_3d}")
                    
                    # Draw bounding box
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put label
                    label = f'{self.model.names[int(cls)]} {conf:.2f}'
                    cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Display depth information
                    cv2.putText(color_image, f'Depth: {depth} mm', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    for i in range(4):
                        cv2.putText(color_image, f"{bbox_3d[i, :]}", corners_2d[i, :], cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 2)

                    detected_boxes.append((box, center_3d, bbox_3d, size_3d))

        if len(detected_boxes) > 0:
            if self.get_parameter("imshow").value:
                cv2.imshow('YOLOv8 RealSense Integration', color_image)
                cv2.waitKey(1)

        img_msg = self.br.cv2_to_imgmsg(color_image)
        img_msg.header = rgbd_msg.header
        self.publisher.publish(img_msg)

        detarray = Detection3DArray()
        detarray.header = rgbd_msg.header
        for box, center_3d, _, size_3d in detected_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf.cpu().numpy()[0]
            cls = box.cls.cpu().numpy()[0]

            det = Detection3D()
            det.header = rgbd_msg.header
            det.bbox.center.position.x = float(center_3d[0])
            det.bbox.center.position.y = float(center_3d[1])
            det.bbox.center.position.z = float(center_3d[2])
            det.bbox.size.x = float(size_3d[0])
            det.bbox.size.y = float(size_3d[1])
            det.bbox.size.z = float(size_3d[2])

            oh = ObjectHypothesisWithPose()
            oh.hypothesis.class_id = str(cls)
            oh.hypothesis.score = float(conf)
            det.results.append(oh)

            det.id = str(cls)

            detarray.detections.append(det)

        self.detection_pub.publish(detarray)


        if self.get_parameter("publish_visualisation_markers").value:
            # Publish markers
            if len(detected_boxes) > 0:
                vmsg = MarkerArray()
                for id ,(box, center_3d, bbox_3d, size_3d) in enumerate(detected_boxes):
                    marker = Marker() 
                    marker.header = rgbd_msg.header
                    marker.ns = "bbox"
                    marker.id = id
                    # marker.type = Marker.LINE_LIST
                    # marker.action = Marker.ADD

                    # marker.scale.x = 1.0  # Line width
                    # marker.scale.y = 1.0  # Line width
                    # marker.scale.z = 1.0  # Line width

                    # marker.color.a = 1.0
                    # marker.color.r = 1.0
                    # marker.color.g = 0.0
                    # marker.color.b = 0.0

                    # # Define the lines of the bounding box
                    # lines = [
                    #     [0, 1], [1, 3], [3, 2], [2, 0],  # Top face
                    #     [4, 5], [5, 7], [7, 6], [6, 4],  # Bottom face
                    #     [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
                    # ]

                    # for line in lines:
                    #     for index in line:
                    #         point = Point()
                    #         point.x = bbox_3d[index, 0]
                    #         point.y = bbox_3d[index, 1]
                    #         point.z = bbox_3d[index, 2]
                    #         marker.points.append(point)

                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD

                    # Position of the center of the cube
                    marker.pose.position.x = float(center_3d[0])/100.0
                    marker.pose.position.y = float(center_3d[1])/100.0
                    marker.pose.position.z = float(center_3d[2])/100.0

                    # Orientation of the cube (no rotation)
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0

                    # Dimensions of the cube
                    marker.scale.x = float(size_3d[0])/100.0
                    marker.scale.y = float(size_3d[1])/100.0
                    marker.scale.z = float(size_3d[2])/100.0

                    # Color and transparency
                    marker.color.a = 0.5  # Semi-transparent
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0

                    vmsg.markers.append(marker)
                
                self.vis_marker_pub.publish(vmsg)





def main(args=None):
      
    # Initialize the rclpy library
    rclpy.init(args=args)
    
    # Create the node
    image_publisher = YoloRealsense()
    
    # Spin the node so the callback function is called.
    rclpy.spin(image_publisher)
    
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    cv2.destroyAllWindows()
    image_publisher.destroy_node()
    
    # Shutdown the ROS client library for Python
    rclpy.shutdown()
    


if __name__ == '__main__':
    main()
