from lerobot.common.robot_devices.motors import feetech_dual_motor_joint
import time

port_number = '/dev/tty.usbmodem58A60699971'

motor_model = 'sts3215'

PRIMARY_ID = 2
SECONDARY_ID = 3

motor_groups={"shoulder_lit": [(PRIMARY_ID, motor_model), (SECONDARY_ID, motor_model)]}

dual_motor_joint_bus = feetech_dual_motor_joint.FeetechMotorsBusV2(port=port_number, motor_groups=motor_groups)
dual_motor_joint_bus.connect()

primary_present_position = dual_motor_joint_bus.read_with_motor_ids([motor_model], [PRIMARY_ID], 'Present_Position')

print('Primary Motor position:')
print(primary_present_position)

print('Secondary Motor position:')
secondary_present_position = dual_motor_joint_bus.read_with_motor_ids([motor_model], [SECONDARY_ID], 'Present_Position')
print(secondary_present_position)

delta_position = 100


primary_target_position = primary_present_position[0] + delta_position

secondary_target_position = secondary_present_position[0] - delta_position

if (primary_target_position > 4095) or (primary_target_position < 0) or (secondary_target_position > 4095) or (secondary_target_position < 0):
    err_msg = f"Delta position out of scope [-2048, 2048], current primary servo poistion is {primary_present_position}, sencondary servo position is {secondary_present_position}, delta position is {delta_position}"
    raise Exception(err_msg)


# set position to 2048
# dual_motor_joint_bus.write_with_motor_ids([motor_model], [PRIMARY_ID, SECONDARY_ID], "Goal_Position", [2048, 2048])

dual_motor_joint_bus.write_with_motor_ids([motor_model], [PRIMARY_ID, SECONDARY_ID], "Goal_Position", [primary_target_position, secondary_target_position])

# 2648,1448

time.sleep(1)
print('=========================')
new_positions = dual_motor_joint_bus.read_with_motor_ids([motor_model], [PRIMARY_ID, SECONDARY_ID], 'Present_Position')

print('Primary position:')
print(new_positions[0])

print('Secondary position:')
print(new_positions[1])

print('+++++++++++++++++++++')
p_coeffs = dual_motor_joint_bus.read_with_motor_ids([motor_model], [PRIMARY_ID, SECONDARY_ID], 'P_Coefficient')
i_coeffs = dual_motor_joint_bus.read_with_motor_ids([motor_model], [PRIMARY_ID, SECONDARY_ID], "I_Coefficient")
d_coeffs = dual_motor_joint_bus.read_with_motor_ids([motor_model], [PRIMARY_ID, SECONDARY_ID], 'D_Coefficient')

mini_start_forces = dual_motor_joint_bus.read_with_motor_ids([motor_model], [PRIMARY_ID, SECONDARY_ID], 'Minimum_Startup_Force')

print('P', p_coeffs)
print('I', i_coeffs)
print('D', d_coeffs)
print('Mini Start Force', mini_start_forces)

