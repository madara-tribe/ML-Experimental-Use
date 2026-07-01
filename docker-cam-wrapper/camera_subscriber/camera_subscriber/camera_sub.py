import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')

        self.bridge = CvBridge()
        self.frame_count = 0

        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info('Camera subscriber started, waiting for images on /image_raw ...')

    def image_callback(self, msg: Image):
        self.frame_count += 1

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            self.get_logger().info(
                f'[RECV OK] frame#{self.frame_count} '
                f'size={frame.shape[1]}x{frame.shape[0]} '
                f'encoding={msg.encoding} '
                f'stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}'
            )

        except CvBridgeError as e:
            self.get_logger().error(
                f'[RECV ERROR] frame#{self.frame_count} CvBridgeError: {e}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
