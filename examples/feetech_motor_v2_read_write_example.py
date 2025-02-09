from lerobot.common.robot_devices.motors import feetech_dual_motor_joint
import time

port_number = '/dev/tty.usbmodem58A60699971'

motor_model = 'sts3215'

SHOULDER_LIFT_PRIMARY_ID = 2
SHOULDER_LIFT_SECONDARY_ID = 3

ELBOW_LIFT_PRIMARY_ID = 4
ELBOW_LIFT_SECONDARY_ID = 5

motor_groups={ 
    "rotation_base": [(1, motor_model)], 
    "shoulder_lift": [(SHOULDER_LIFT_PRIMARY_ID, motor_model), (SHOULDER_LIFT_SECONDARY_ID, motor_model)],
    "elbow_lift": [(ELBOW_LIFT_PRIMARY_ID, motor_model), (ELBOW_LIFT_SECONDARY_ID, motor_model)]
}

dual_motor_joint_bus = feetech_dual_motor_joint.FeetechMotorsBusV2(port=port_number, motor_groups=motor_groups)
dual_motor_joint_bus.connect()

values = dual_motor_joint_bus.read('Present_Position')

# Read 
print(values)

new_position = values.copy()

new_position[1] = new_position[1] - 200
print('write.....', new_position)
dual_motor_joint_bus.write('Goal_Position', new_position)
