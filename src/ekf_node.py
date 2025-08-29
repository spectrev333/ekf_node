#!/usr/bin/env python3
import sys
import rospy
from EKF import EKF
from geometry_msgs.msg import Pose2D
from driverless_msgs.msg import PoseStamped, velocity
from sensor_msgs.msg import Imu
import numpy as np

class EKFNode:
    
    def __init__(self, freq=100):
        rospy.init_node("ekf_node")
        rospy.loginfo("ekf_node started...")

        self.freq = freq

        # Covariance matrices already initialized in 
        self.ekf = EKF(1.5)
        # Can be modified here:
        # ekf.Q = np.diag([1e-3, 1e-3, 2e-2, 2e-1, 1e-5, 1e-5])  # process noise [px, py, theta, v, ba, bw]
        # ekf.R_tone = np.diag([0.01]) # Tone (sigma ~0.5m/s)
        # ekf.R_slam = np.diag([0.22, 0.22, 0.1]) # SLAM ( 0.1-10m , 0.001-0.01rad)

        # Last measurment from each sensor
        self.last_imu_meas = None
        self.last_slam_meas = None
        self.last_tone_meas = None

        self.last_slam_meas_time = rospy.Time(0)
        self.last_tone_meas_time = rospy.Time(0)

        # Bookkeeping to avoid reusing measures
        self.last_update_time = rospy.Time.now()
        self.last_slam_processed_time = rospy.Time(0)
        self.last_tone_processed_time = rospy.Time(0)

        self.has_initial_imu = False
        self.has_initial_slam = False
        self.has_initial_tone = False

        rospy.Subscriber('/orb_slam3/camera_pose', PoseStamped, self.slam_callback)
        rospy.Subscriber('/zed2i/zed_node/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('/data_logger/speed_actual', velocity, self.tone_callback)

        self.pose_pub = rospy.Publisher('/ekf_node/pose', Pose2D, queue_size=1)

    def slam_callback(self, msg):
        self.last_slam_meas = np.array([msg.pose.position.x, msg.pose.position.y, msg.yaw])
        self.last_slam_meas_time = msg.header.stamp
        self.has_initial_slam = True

    def tone_callback(self, msg):
        self.last_tone_meas = np.array([msg.velocity])
        self.last_tone_meas_time = msg.header.stamp
        self.has_initial_tone = True

    def imu_callback(self, msg):
        self.last_imu_meas = np.array([msg.linear_acceleration.x, msg.angular_velocity.z])
        self.has_initial_imu = True
    
    def run(self):
        rate = rospy.Rate(self.freq)
        while not rospy.is_shutdown():
            # Wait for all sensors
            if self.has_initial_imu and self.has_initial_slam and self.has_initial_tone:
                #current_time = rospy.Time.now()
                #dt = (current_time - self.last_update_time).to_sec()
                dt = 1 / 100 # TODO: da calcolare con ros magari

                # Prediction step
                self.ekf.predict(self.last_imu_meas, dt)

                # Update step using SLAM
                if self.last_slam_meas_time > self.last_slam_processed_time:
                    self.ekf.update_slam(self.last_slam_meas)
                    self.last_slam_processed_time = self.last_slam_meas_time

                # Update step using tone wheels
                if self.last_tone_meas_time > self.last_tone_processed_time:
                    self.ekf.update_tone_wheels(self.last_tone_meas)
                    self.last_tone_processed_time = self.last_tone_meas_time

                new_pose = Pose2D()
                new_pose.x = self.ekf.x[0]
                new_pose.y = self.ekf.x[1]
                new_pose.theta = self.ekf.x[2]
                self.pose_pub.publish(new_pose)

            rate.sleep()


if __name__ == '__main__':
    try:
        ekf = EKFNode(freq=100)
        ekf.run()
    except KeyboardInterrupt:
        pass
