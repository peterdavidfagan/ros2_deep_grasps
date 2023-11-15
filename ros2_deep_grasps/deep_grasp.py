"""A service for deep grasp detection."""

from ros2_deep_grasps_interfaces.srv import GraspPose
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

import onnx
import onnxruntime as rt

import rclpy
from rclpy.node import Node

class GRCNN(Node):

    def __init__(self):
        super().__init__('grcnn_grasp_pose_service')
        self.logger = self.get_logger()

        self.grcnn = rt.InferenceSession("grcnn.onnx") # TODO: read path from config

        # ensure parallel execution of camera callbacks
        # we want to get images while also executing the calibration service
        self.camera_callback_group = ReentrantCallbackGroup()

        # set QOS profile for camera image callback
        self.camera_qos_profile = QoSProfile(
                depth=1,
                history=QoSHistoryPolicy(rclpy.qos.HistoryPolicy.KEEP_LAST),
                reliability=QoSReliabilityPolicy(rclpy.qos.ReliabilityPolicy.RELIABLE),
            )

        # create a bridge between ROS2 and OpenCV
        self.cv_bridge = CvBridge()

        # subscribe to camera image
        self.create_subscription(
            Image,
            "", # TODO: read from config
            self._image_callback,
            self.camera_qos_profile,
            callback_group=self.camera_callback_group,
        )

        # create service for requesting grasp pose
        self.srv = self.create_service(
                GraspPose, 
                'get_grcnn_grasp_pose', 
                self.grasp_pose_callback,
                )
        
        # track latest image
        self._last_image = None


    def _image_callback(self, msg):
        """Callback function for image topic"""
        self._latest_image = msg

    def grasp_pose_callback(self, request, response):
        self.get_logger().info('Computing grasp pose...')

        # preprocess the latest RGBD image for the GRCNN
        img = self.cv_bridge.imgmsg_to_cv2(self._latest_image, "rgba8") # check encoding


        # run the image through the GRCNN using onnxruntime
        outputs = self.grcnn.run(img) # TODO: check input format

        # format response based on GRCNN output
        
        
        return response

def main(args=None):
    rclpy.init(args=args)
    grasp_pose_service = GraspPoseService()
    rclpy.spin(grasp_pose_service)
    grasp_pose_service.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
