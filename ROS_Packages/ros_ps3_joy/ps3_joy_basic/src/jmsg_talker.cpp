#include <ros/ros.h>
#include <ps3_joy_msgs/Custom.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "jmsg_talker");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<ps3_joy_msgs::Custom>("chatter", 10);

  ros::Rate loop_rate(1);
  while (ros::ok())
  {
    ps3_joy_msgs::Custom data;
    data.word = "hello";
    data.number = 10;
    pub.publish(data);
    ros::spinOnce();
    loop_rate.sleep();
  }
}
