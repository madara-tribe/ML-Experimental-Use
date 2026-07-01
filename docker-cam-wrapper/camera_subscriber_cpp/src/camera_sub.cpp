#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class CameraSubscriber : public rclcpp::Node
{
public:
  CameraSubscriber()
  : Node("camera_subscriber")
  {
    sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw",
      rclcpp::SensorDataQoS(),
      std::bind(&CameraSubscriber::image_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Camera subscriber started, waiting for images...");
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
      RCLCPP_INFO(this->get_logger(),
        "Received image: %dx%d", frame.cols, frame.rows);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraSubscriber>());
  rclcpp::shutdown();
  return 0;
}
