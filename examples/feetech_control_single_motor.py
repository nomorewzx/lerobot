from lerobot.common.robot_devices.motors import feetech_motor_group
import time

from lerobot.common.robot_devices.motors.configs import FeetechMotorGroupsBusConfig
port_number = '/dev/tty.usbmodem58A60699971'

motor_model = 'sts3215'

PRIMARY_ID = 6

motor_groups={"gripper_moving": [(PRIMARY_ID, motor_model)]}
cfg = FeetechMotorGroupsBusConfig(port=port_number, motors=motor_groups)

dual_motor_joint_bus = feetech_motor_group.FeetechMotorGroupsBus(config=cfg)

dual_motor_joint_bus.connect()

primary_present_position = dual_motor_joint_bus.read_with_motor_ids([motor_model], [PRIMARY_ID], 'Present_Position')

abs_pos = 2048

dual_motor_joint_bus.write_with_motor_ids([motor_model], [PRIMARY_ID], "Goal_Position", [abs_pos])

# dual_motor_joint_bus.write_with_motor_ids([motor_model], [PRIMARY_ID], "P_Coefficient", [10])

# dual_motor_joint_bus.write_with_motor_ids([motor_model], [PRIMARY_ID], "Minimum_Startup_Force", [6])

# dual_motor_joint_bus.write_with_motor_ids([motor_model], [PRIMARY_ID], "Goal_Position", [2048])

print('Primary Motor position:')
print(primary_present_position)