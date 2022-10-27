#! /usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
class KeepMovement(object):
    def __init__(self):
        self.ahead_range = 1 
        # for sensor sub
        self.sensor_sub = rospy.Subscriber('jetbot/laser_scan', LaserScan, self.scan_callback) 
        #self.sensor_sub = rospy.Subscriber('/imu_sensor/data', Imu, self.scan_callback)   
        # for topic pub
        self.cmdvel_pub = rospy.Publisher('jetbot/diff_drive_controller/cmd_vel', Twist, queue_size = 1)
        self.state_change_time = rospy.Time.now()    # current time
        self.driving_forward = True  # go ahead is 
        self.rate = rospy.Rate(10)   #10Hz
        self.movement_loop()

    def update(self):
        twist = Twist()
        # can go ahead
        if self.driving_forward:
            twist.linear.x = -1
            twist.angular.z = 0
        # can go ahead and rotate 
        else:
            twist.linear.x = 0
            twist.angular.z = 1
        self.cmdvel_pub.publish(twist)  
        self.rate.sleep()   
        
    def scan_callback(self, msg):
        # get distance from obstacle
        self.ahead_range = msg.ranges[len(msg.ranges)//2]

    def movement_loop(self):

        while not rospy.is_shutdown():

            # go ahead
            if self.driving_forward:
                if (self.ahead_range < 1.0 or rospy.Time.now() > self.state_change_time):
                    self.driving_forward = False 
                    # rotate 5 duration 
                    self.state_change_time = rospy.Time.now() + rospy.Duration(20)

            # not go ahead
            else:
                # wait until ratate direction
                if rospy.Time.now() > self.state_change_time:
                    self.driving_forward = True  
                    self.state_change_time = rospy.Time.now() + rospy.Duration(5)   
            
            # update parameter
            self.update()

if __name__=='__main__':
    rospy.init_node('jetbot_drive')
    keep_move = KeepMovement()
    rospy.spin() 
