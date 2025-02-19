from lerobot.common.robot_devices.motors import feetech_motor_pingti
import time

port_number = '/dev/tty.usbmodem58A60699971'

motor_model = 'sts3215'

MOTOR_ID = 1


motor_groups={"shoulder_lit": [(MOTOR_ID, motor_model)]}

dual_motor_joint_bus = feetech_motor_pingti.FeetechMotorsBusV2(port=port_number, motor_groups=motor_groups)
dual_motor_joint_bus.connect()

present_position = dual_motor_joint_bus.read_with_motor_ids([motor_model], [MOTOR_ID], 'Present_Position')

print('Motor position:')
print(present_position)

delta_position = 2000

target_position = present_position[0] + delta_position

if (target_position > 4095) or (target_position < 0):
    err_msg = f"Delta position out of scope [-2048, 2048], current primary servo poistion is {present_position}, delta position is {delta_position}"
    raise Exception(err_msg)


# set position to 2048
# dual_motor_joint_bus.write_with_motor_ids([motor_model], [MOTOR_ID], "Goal_Position", [1024])

dual_motor_joint_bus.write_with_motor_ids([motor_model], [MOTOR_ID], "Goal_Position", [target_position])
import time

time.sleep(1)
print('=========================')
new_position = dual_motor_joint_bus.read_with_motor_ids([motor_model], [MOTOR_ID], 'Present_Position')

print('Motor position:')
print(new_position)

print('+++++++')
mini_start_forces = dual_motor_joint_bus.read_with_motor_ids([motor_model], [MOTOR_ID], 'Minimum_Startup_Force')
print('Mini Start Force', mini_start_forces)

