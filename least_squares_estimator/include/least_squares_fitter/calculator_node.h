#ifndef DISTANCE_ESTIMATOR_NODE_HPP_
#define DISTANCE_ESTIMATOR_NODE_HPP_

#include "rclcpp/rclcpp.hpp"
#include <cmath>
#include <algorithm>
#include "std_msgs/msg/float64.hpp"

class DistanceEstimatorNode : public rclcpp::Node {
public:
    explicit DistanceEstimatorNode(const rclcpp::NodeOptions & options);
private:
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr publisher_;
    double max_x = 100.0;
    double quadraticFunction(double x);
    double calculateClampedAngle(double x, double z, double max_x);
};

#endif  // DISTANCE_ESTIMATOR_NODE_HPP_

