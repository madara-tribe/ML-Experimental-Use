#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <stdexcept>

class CameraPublisher : public rclcpp::Node
{
public:
  CameraPublisher()
  : Node("camera_publisher")
  {
    this->declare_parameter<std::string>("device_path", "/dev/video0");
    const std::string device_path = this->get_parameter("device_path").as_string();

    cap_.open(device_path);
    cap_.set(cv::CAP_PROP_FPS, 30);
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    if (!cap_.isOpened()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open camera at %s", device_path.c_str());
      throw std::runtime_error("Cannot open camera: " + device_path);
    }

    pub_ = this->create_publisher<sensor_msgs::msg::Image>(
      "/image_raw", rclcpp::SensorDataQoS());

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(33),
      std::bind(&CameraPublisher::timer_callback, this));

    RCLCPP_INFO(this->get_logger(), "Camera opened at %s", device_path.c_str());
  }

  ~CameraPublisher()
  {
    if (cap_.isOpened()) {
      cap_.release();
      RCLCPP_INFO(this->get_logger(), "Camera released.");
    }
  }

private:
  void timer_callback()
  {
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      RCLCPP_WARN(this->get_logger(), "Failed to capture frame");
      return;
    }
    auto msg = cv_bridge::CvImage(
      std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
    msg->header.stamp = this->now();
    pub_->publish(*msg);
  }

  cv::VideoCapture cap_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<CameraPublisher>());
  } catch (const std::runtime_error & e) {
    RCLCPP_ERROR(rclcpp::get_logger("main"), "%s", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
