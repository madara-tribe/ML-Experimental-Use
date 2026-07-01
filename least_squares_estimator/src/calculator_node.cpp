#include "least_squares_fitter/calculator_node.h"

DistanceEstimatorNode::DistanceEstimatorNode(const rclcpp::NodeOptions & options)
: Node("least_squares_fitter", options)
{
    double relative_dist = 234;  // depth median
    double real_x = 23.11; // cm
    double abs_dist = quadraticFunction(relative_dist);
    double angle_deg = calculateClampedAngle(real_x, abs_dist, max_x);

    RCLCPP_INFO(this->get_logger(), "Estimated Distance: %.2f cm, Servo Angle: %.2f deg", abs_dist, angle_deg);

    publisher_ = this->create_publisher<std_msgs::msg::Float64>("hw_px3", 10);
    std_msgs::msg::Float64 msg;
    msg.data = angle_deg;
    publisher_->publish(msg);
}

    double DistanceEstimatorNode::quadraticFunction(double x) 
    {
        double a = 0.00143026;
        double b = 0.32379224;
        double c = -56.7942361;
        return a * std::pow(x, 2) + b * x + c;
    }

    double DistanceEstimatorNode::calculateClampedAngle(double x, double z, double max_x) {
        // Clamp x to the range [0, max_x]
        x = std::max(0.0, std::min(x, max_x));
        // coordinates
	double Ax = 0.0, Ay = 0.0;
        double Px = max_x / 2.0, Py = z;
        double hx = x, hy = 0.0;

        // Vectors AP and HP (origin shifted to point P)
        double vec_AP_x = Ax - Px;
        double vec_AP_y = Ay - Py;

        double vec_hP_x = hx - Px;
        double vec_hP_y = hy - Py;

        // Dot product and vector magnitude
        double dot = vec_AP_x * vec_hP_x + vec_AP_y * vec_hP_y;
        double mag_AP = std::hypot(vec_AP_x, vec_AP_y);
        double mag_hP = std::hypot(vec_hP_x, vec_hP_y);

	// Safety check to avoid division by zero
        if (mag_AP == 0 || mag_hP == 0) return 0.0;

        // cosθ = dot / (|a||b|)
        double cos_theta = dot / (mag_AP * mag_hP);

        // Clamp cos_theta to [-1, 1] to prevent domain errors
        if (cos_theta < -1.0) cos_theta = -1.0;
        if (cos_theta > 1.0) cos_theta = 1.0;

        double angle_rad = std::acos(cos_theta);
        return angle_rad * (180.0 / M_PI);
    }

// main function similar to inference.cpp
int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DistanceEstimatorNode>(rclcpp::NodeOptions{});
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

