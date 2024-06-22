import launch
import launch_ros.actions
import os

from ament_index_python.packages import get_package_share_directory
from launch.conditions import LaunchConfigurationEquals
from launch.actions import OpaqueFunction


def get_node(context):
    config_file = launch.substitutions.LaunchConfiguration('config_file').perform(context)

    vms_bridge = launch_ros.actions.Node(
        package='yolo8_realsense',
        executable='detector',
        name='flower_detector',
        namespace=launch.substitutions.LaunchConfiguration("namespace").perform(context),
        parameters=[config_file, 
                    {"model_path": launch.substitutions.LaunchConfiguration("model").perform(context)}]
    )

    return [vms_bridge]


def generate_launch_description():

    namespace_launch_arg = launch.actions.DeclareLaunchArgument("namespace", default_value="")
    
    params_file = os.path.join(get_package_share_directory('yolo8_realsense'), 'config', 'detector_params.yaml')
    config_file_launch_arg = launch.actions.DeclareLaunchArgument(
        'config_file', default_value=params_file
    )

    model_file = os.path.join(get_package_share_directory('yolo8_realsense'), 'model', 'best.pt')
    model_file_launch_arg = launch.actions.DeclareLaunchArgument(
        'model', default_value=model_file
    )

    return launch.LaunchDescription([
        namespace_launch_arg,
        config_file_launch_arg,
        model_file_launch_arg,
        OpaqueFunction(function=get_node)    
    ])
