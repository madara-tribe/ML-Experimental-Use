from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # --- Arguments ---
    backend_arg = DeclareLaunchArgument(
        'backend',
        default_value='python',
        description='Backend to use: "python" or "cpp"'
    )

    device_path_arg = DeclareLaunchArgument(
        'device_path',
        default_value='/dev/video0',
        description='Camera device path (e.g. /dev/video0)'
    )

    backend = LaunchConfiguration('backend')
    device_path = LaunchConfiguration('device_path')

    is_python = PythonExpression(["'", backend, "' == 'python'"])
    is_cpp    = PythonExpression(["'", backend, "' == 'cpp'"])

    # --- Python nodes ---
    python_publisher = Node(
        condition=IfCondition(is_python),
        package='camera_publisher',
        executable='camera_pub',
        name='camera_publisher',
        output='screen',
        parameters=[{'device_path': device_path}]
    )

    python_subscriber = Node(
        condition=IfCondition(is_python),
        package='camera_subscriber',
        executable='camera_sub',
        name='camera_subscriber',
        output='screen'
    )

    # --- C++ nodes ---
    cpp_publisher = Node(
        condition=IfCondition(is_cpp),
        package='camera_publisher_cpp',
        executable='camera_pub',
        name='camera_publisher',
        output='screen',
        parameters=[{'device_path': device_path}]
    )

    cpp_subscriber = Node(
        condition=IfCondition(is_cpp),
        package='camera_subscriber_cpp',
        executable='camera_sub',
        name='camera_subscriber',
        output='screen'
    )

    return LaunchDescription([
        backend_arg,
        device_path_arg,
        python_publisher,
        python_subscriber,
        cpp_publisher,
        cpp_subscriber,
    ])
