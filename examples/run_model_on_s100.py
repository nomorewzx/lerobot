import time
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera


inference_time_s = 50
fps = 30
device = "cpu"  # TODO: On Mac, use "mps" or "cpu"

ckpt_path = "outputs/train/act_so100_test_20241214/checkpoints/080000/pretrained_model"
policy = ACTPolicy.from_pretrained(ckpt_path)
policy.to(device)

follower_port = "/dev/tty.usbserial-A50285BI"
leader_port = "/dev/tty.usbmodem58A60699971"

motor_model = "sts3215"

follower_arm = FeetechMotorsBus(
    port=follower_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, motor_model),
        "shoulder_lift": (2, motor_model),
        "elbow_flex": (3, motor_model),
        "wrist_flex": (4, motor_model),
        "wrist_roll": (5, motor_model),
        "gripper": (6, motor_model),
    },
)


leader_arm = FeetechMotorsBus(
    port=leader_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, motor_model),
        "shoulder_lift": (2, motor_model),
        "elbow_flex": (3, motor_model),
        "wrist_flex": (4, motor_model),
        "wrist_roll": (5, motor_model),
        "gripper": (6, motor_model),
    },
)


robot = ManipulatorRobot(
    robot_type="so100",
    leader_arms={"main": leader_arm},
    follower_arms={"main": follower_arm},
    calibration_dir=".cache/calibration/so100",
    cameras={
        "front_camera": OpenCVCamera(0, fps=30, width=640, height=480),
        "top_camera": OpenCVCamera(1, fps=30, width=640, height=480),
    },
)

robot.connect()

follower_pos = robot.follower_arms["main"].read("Present_Position")

print('Start position++++++++++++', follower_pos)

for _ in range(inference_time_s * fps):
    start_time = time.perf_counter()

    # Read the follower state and access the frames from the cameras
    observation = robot.capture_observation()

    # Convert to pytorch format: channel first and float32 in [0,1]
    # with batch dimension
    for name in observation:
        if "image" in name:
            observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].permute(2, 0, 1).contiguous()
        observation[name] = observation[name].unsqueeze(0)
        observation[name] = observation[name].to(device)

    # Compute the next action with the policy
    # based on the current observation
    action = policy.select_action(observation)
    # Remove batch dimension
    action = action.squeeze(0)
    # Move to cpu, if not already the case
    action = action.to("cpu")
    print(action)
    # Order the robot to move
    robot.send_action(action)
    
    print(action)
    
    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)