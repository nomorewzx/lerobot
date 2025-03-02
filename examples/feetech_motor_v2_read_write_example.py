from lerobot.common.robot_devices.motors import feetech_motor_group
import time

from lerobot.common.robot_devices.motors.configs import FeetechMotorGroupsBusConfig

port_number = '/dev/tty.usbmodem58A60699971'

port_number_2 = '/dev/tty.usbserial-A50285BI'

motor_model = 'sts3215'

BASE_YAW_ID = 1
SHOULDER_LIFT_PRIMARY_ID = 2
SHOULDER_LIFT_SECONDARY_ID = 3

ELBOW_LIFT_PRIMARY_ID = 4
ELBOW_LIFT_SECONDARY_ID = 5

WRIST_LIFT_ID = 6
WRIST_ROLL_ID = 7
GRIPPER_MOVING_ID = 8

motor_groups={ 
    "rotation_base": [(1, motor_model)], 
    "shoulder_lift": [(SHOULDER_LIFT_PRIMARY_ID, motor_model), (SHOULDER_LIFT_SECONDARY_ID, motor_model)],
    "elbow_lift": [(ELBOW_LIFT_PRIMARY_ID, motor_model), (ELBOW_LIFT_SECONDARY_ID, motor_model)],
    "wrist_lift": [(WRIST_LIFT_ID, motor_model)],
    "wrist_roll": [(WRIST_ROLL_ID, motor_model)],
    # "gripper_moving": [(GRIPPER_MOVING_ID, motor_model)]
}

cfg = FeetechMotorGroupsBusConfig(port=port_number_2, motors=motor_groups)

dual_motor_joint_bus = feetech_motor_group.FeetechMotorGroupsBus(config=cfg)
dual_motor_joint_bus.connect()

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


# dual_motor_joint_bus.write_with_motor_ids([motor_model], [BASE_YAW_ID], "Minimum_Startup_Force", [5])
# dual_motor_joint_bus.write_with_motor_ids([motor_model], [SHOULDER_LIFT_PRIMARY_ID], "Minimum_Startup_Force", [10])
# dual_motor_joint_bus.write_with_motor_ids([motor_model], [SHOULDER_LIFT_SECONDARY_ID], "Minimum_Startup_Force", [10])
# dual_motor_joint_bus.write_with_motor_ids([motor_model], [ELBOW_LIFT_PRIMARY_ID], "Minimum_Startup_Force", [10])
# dual_motor_joint_bus.write_with_motor_ids([motor_model], [ELBOW_LIFT_SECONDARY_ID], "Minimum_Startup_Force", [10])

# dual_motor_joint_bus.write_with_motor_ids([motor_model], [BASE_YAW_ID], "P_Coefficient", [6])
# dual_motor_joint_bus.write_with_motor_ids([motor_model], [SHOULDER_LIFT_PRIMARY_ID], "P_Coefficient", [6])
# dual_motor_joint_bus.write_with_motor_ids([motor_model], [SHOULDER_LIFT_SECONDARY_ID], "P_Coefficient", [6])
# dual_motor_joint_bus.write_with_motor_ids([motor_model], [ELBOW_LIFT_PRIMARY_ID], "P_Coefficient", [10])
# dual_motor_joint_bus.write_with_motor_ids([motor_model], [ELBOW_LIFT_SECONDARY_ID], "P_Coefficient", [10])


values = dual_motor_joint_bus.read('Present_Position')

# Read 
print(values)

# new_position = values.copy()

# new_position[0] = new_position[0] - 200
# new_position[1] = new_position[1] - 200
# new_position[2] = new_position[2] + 200
# new_position[3] = new_position[3] + 200
# new_position[4] = new_position[4] + 200
# new_position[5] = new_position[5] + 200

# Write abs value
from positions_example import positions, shoulder_rotate_90_pos
all_motor_ids = [1,[2,3],[4,5],6,7,8]

for pos in shoulder_rotate_90_pos:
    joint_num = 5
    new_pos = pos[:joint_num]
    valid_motor_ids = flatten_list(all_motor_ids[:joint_num])
    print('write.....', new_pos)
    dual_motor_joint_bus.write('Goal_Position', new_pos)
    position_after_execution = dual_motor_joint_bus.read('Present_Position')
    delta_gap = new_pos - position_after_execution
    print('Delta Gap', delta_gap)

    load_values = dual_motor_joint_bus.read_with_motor_ids([motor_model] * len(valid_motor_ids), valid_motor_ids, "Present_Load")
    print('---------- load ', load_values)
    voltage_values = dual_motor_joint_bus.read_with_motor_ids([motor_model] * len(valid_motor_ids), valid_motor_ids, "Present_Voltage")
    print('---------- voltage ', voltage_values)
    current_values = dual_motor_joint_bus.read_with_motor_ids([motor_model] * len(valid_motor_ids), valid_motor_ids, "Present_Current")
    print('---------- current ', current_values)
