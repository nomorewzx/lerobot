from lerobot.common.robot_devices.motors import feetech_motor_pingti
import time

port_number = '/dev/tty.usbmodem58A60699971'

motor_model = 'sts3215'

DUAL_SERVO_JOINT_CONTROL_MODES = ['delta', 'abs_degree']

CURRENT_CONTROL_MODE = DUAL_SERVO_JOINT_CONTROL_MODES[1]

SHOULDER_LIFT_PRIMARY_ID = 2
SHOULDER_LIFT_SECONDARY_ID = 3

ELBOW_LIFT_PRIMARY_ID = 4
ELBOW_LIFT_SECONDARY_ID = 5

motor_groups={ 
    "rotation_base": [(1, motor_model)], 
    "shoulder_lift": [(SHOULDER_LIFT_PRIMARY_ID, motor_model), (SHOULDER_LIFT_SECONDARY_ID, motor_model)],
    "elbow_lift": [(ELBOW_LIFT_PRIMARY_ID, motor_model), (ELBOW_LIFT_SECONDARY_ID, motor_model)]
}

dual_motor_joint_bus = feetech_motor_pingti.FeetechMotorsBusV2(port=port_number, motor_groups=motor_groups)
dual_motor_joint_bus.connect()

values = dual_motor_joint_bus.read('Present_Position')

print(values)

primary_present_position = dual_motor_joint_bus.read_with_motor_ids([motor_model], [SHOULDER_LIFT_PRIMARY_ID], 'Present_Position')

print('Primary Motor position:')
print(primary_present_position)

print('Secondary Motor position:')
secondary_present_position = dual_motor_joint_bus.read_with_motor_ids([motor_model], [SHOULDER_LIFT_SECONDARY_ID], 'Present_Position')
print(secondary_present_position)

if CURRENT_CONTROL_MODE == DUAL_SERVO_JOINT_CONTROL_MODES[1]:
    primary_target_degree = 300
    print('primary target degree.....', primary_target_degree)

    primary_target_positions = feetech_motor_pingti.convert_degrees_to_steps(primary_target_degree, ['sts3215'])

    primary_target_position = primary_target_positions[0]
    secondary_target_position = 4095 - primary_target_position

    print('Primary target_position', primary_target_position)
    print('Secondary target position', secondary_target_position)
else:
    delta_position = 270
    primary_target_position = primary_present_position[0] + delta_position

    secondary_target_position = secondary_present_position[0] - delta_position


# primary_target_position = primary_present_position[0] + delta_position

# secondary_target_position = secondary_present_position[0] - delta_position

if (primary_target_position > 4095) or (primary_target_position < 0) or (secondary_target_position > 4095) or (secondary_target_position < 0):
    err_msg = f"Delta position out of scope [-2048, 2048], current primary servo poistion is {primary_present_position}, sencondary servo position is {secondary_present_position}, delta position is"
    raise Exception(err_msg)


# set position to 2048
# dual_motor_joint_bus.write_with_motor_ids([motor_model], [PRIMARY_ID, SECONDARY_ID], "Goal_Position", [2048, 2048])

dual_motor_joint_bus.write_with_motor_ids([motor_model], [SHOULDER_LIFT_PRIMARY_ID, SHOULDER_LIFT_SECONDARY_ID], "Goal_Position", [primary_target_position, secondary_target_position])

# 2648,1448

time.sleep(1)
print('=========================')
new_positions = dual_motor_joint_bus.read_with_motor_ids([motor_model], [SHOULDER_LIFT_PRIMARY_ID, SHOULDER_LIFT_SECONDARY_ID], 'Present_Position')

print('Primary position:')
print(new_positions[0])

print('Secondary position:')
print(new_positions[1])