from lerobot.common.robot_devices.motors import feetech
import time
from pynput import keyboard


gripper_sensitivity = 50   # Angle for the gripper to open

port_number = '/dev/tty.usbserial-A50285BI'

motor_model = 'sts3215'

GRIPPER_ID = 6

motors={"gripper": (GRIPPER_ID, motor_model)}

gripper_motor = feetech.FeetechMotorsBus(port=port_number, motors=motors)
gripper_motor.connect()

# enable torque
gripper_motor.write_with_motor_ids([motor_model], [GRIPPER_ID], "Torque_Enable", feetech.TorqueMode.ENABLED.value)
print('Torque mode enabled')



def convert_present_load_value_to_correct_number(value):
    """Convert a 16-bit integer to signed."""
    higher_byte_value = (value >> 8) & 0xFF
    lower_byte_value = value & 0xFF 

    if higher_byte_value == 4:
        return -lower_byte_value
    return lower_byte_value

# Function to open the gripper
def open_gripper():
    present_load = gripper_motor.read_with_motor_ids([motor_model], [GRIPPER_ID], 'Present_Load')
    # present_load is signed int, need to convert raw readings (unsigned int) from servo to signed int
    
    present_load = convert_present_load_value_to_correct_number(present_load[0])
    print('Present load is', present_load)
    
    # read present position
    present_position = gripper_motor.read_with_motor_ids([motor_model], [GRIPPER_ID], 'Present_Position')

    print("Opening gripper...")
    gripper_motor.write_with_motor_ids([motor_model], [GRIPPER_ID], "Goal_Position", present_position[0]+gripper_sensitivity)

    present_position = gripper_motor.read_with_motor_ids([motor_model], [GRIPPER_ID], 'Present_Position')
    print('Present position after moving:', present_position)
    time.sleep(0.1)

# Function to close the gripper
def close_gripper():
    object_torque_limit_mapping = {
        'kiwi': 60,
        'paper_cup': 400,
        'cherry_tomato': 30
    }

    OBJECT_TO_GRIP = 'paper_cup'

    gripper_motor.write_with_motor_ids([motor_model], [GRIPPER_ID], "Torque_Limit", object_torque_limit_mapping[OBJECT_TO_GRIP])

    final_torque_limit = gripper_motor.read_with_motor_ids([motor_model], [GRIPPER_ID], 'Torque_Limit')

    print(f'Grip {OBJECT_TO_GRIP}, set torque limit as {final_torque_limit}')

    print("Closing gripper...")
    present_load = gripper_motor.read_with_motor_ids([motor_model], [GRIPPER_ID], 'Present_Load')
    
    present_load = convert_present_load_value_to_correct_number(present_load[0])
    print('Present load is', present_load)
    

    # read present position
    present_position = gripper_motor.read_with_motor_ids([motor_model], [GRIPPER_ID], 'Present_Position')

    gripper_motor.write_with_motor_ids([motor_model], [GRIPPER_ID], "Goal_Position", present_position[0]-gripper_sensitivity)

    present_position = gripper_motor.read_with_motor_ids([motor_model], [GRIPPER_ID], 'Present_Position')
    print('Present position after moving:', present_position)
    time.sleep(0.05)

def close_gripper_with_impedance_control():
    load_threshold = 80
    max_torque_limit = 300
    while True:
        # 读取当前负载
        present_load = gripper_motor.read_with_motor_ids([motor_model], [GRIPPER_ID], 'Present_Load')
        present_load = convert_present_load_value_to_correct_number(present_load[0])
        print('Present load:', present_load)

        # 读取当前位置
        present_position = gripper_motor.read_with_motor_ids([motor_model], [GRIPPER_ID], 'Present_Position')[0]

        if abs(present_load) > load_threshold:
            print("Force threshold exceeded, adjusting torque and stopping further closure.")

            # 如果负载超出阈值，减少扭矩限制
            gripper_motor.write_with_motor_ids([motor_model], [GRIPPER_ID], "Torque_Limit", int(max_torque_limit * 0.5))

            # 停止进一步关闭，轻微反弹以缓解力
            new_position = present_position + int(gripper_sensitivity * 0.1)  # 轻微松开
            gripper_motor.write_with_motor_ids([motor_model], [GRIPPER_ID], "Goal_Position", new_position)
            break

        else:
            print("Force within threshold, continuing closure.")

            # 负载在安全范围内，保持默认扭矩限制
            gripper_motor.write_with_motor_ids([motor_model], [GRIPPER_ID], "Torque_Limit", max_torque_limit)

            # 继续向关闭方向移动
            new_position = present_position - gripper_sensitivity
            gripper_motor.write_with_motor_ids([motor_model], [GRIPPER_ID], "Goal_Position", new_position)

        # 打印当前位置以跟踪运动
        present_position = gripper_motor.read_with_motor_ids([motor_model], [GRIPPER_ID], 'Present_Position')[0]
        print('Present position after moving:', present_position)

        # 延时避免过快操作
        time.sleep(0.2)



# Define keyboard event handlers
def on_press(key):
    try:
        if key == keyboard.Key.up:  # Arrow up to open gripper
            open_gripper()
        elif key == keyboard.Key.down:  # Arrow down to close gripper
            close_gripper_with_impedance_control()
    except Exception as e:
        print(f"Error: {e}")

def on_release(key):
    if key == keyboard.Key.esc:  # Exit the program on Esc key
        print("Exiting...")
        gripper_motor.disconnect()
        return False

# Main program to listen for keyboard input
print("Press UP arrow to open the gripper, DOWN arrow to close it, and ESC to exit.")
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
