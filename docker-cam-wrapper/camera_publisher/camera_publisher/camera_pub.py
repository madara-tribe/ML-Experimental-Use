import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        self.declare_parameter('device_path', '/dev/video0')
        device_path = self.get_parameter('device_path').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(device_path)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera at {device_path}')
            raise RuntimeError(f'Cannot open camera: {device_path}')

        self.pub = self.create_publisher(Image, '/image_raw', qos_profile_sensor_data)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)
        self.get_logger().info(f'Camera opened at {device_path}')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
            self.get_logger().info('Camera released.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraPublisher()
        rclpy.spin(node)
    except RuntimeError as e:
        print(f'[ERROR] {e}')
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
