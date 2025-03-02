so100.yaml

Find port:
python lerobot/scripts/find_motors_bus_port.py


Calibrate motors:

python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem58A60699971 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1


python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem58A60699971 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 2


python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem58A60699971 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 3

python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem58A60699971 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 4

python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem58A60699971 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 5

python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem58A60699971 \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 6




# Follower port:

/dev/tty.usbserial-A50285BI


python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbserial-A50285BI \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 1


python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbserial-A50285BI \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 2

python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbserial-A50285BI \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 3

  python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbserial-A50285BI \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 4

  python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbserial-A50285BI \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 5

python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbserial-A50285BI \
  --brand feetech \
  --model sts3215 \
  --baudrate 1000000 \
  --ID 6


### calibrate follower & leader arm

python lerobot/scripts/control_robot.py calibrate \
    --robot-path lerobot/configs/robot/so100.yaml \
    --robot-overrides '~cameras' --arms main_follower


### Teleoperate without camera
python lerobot/scripts/control_robot.py teleoperate \
    --robot-path lerobot/configs/robot/so100.yaml \
    --robot-overrides '~cameras' \
    --display-cameras 0

## Visualize cameras
python lerobot/common/robot_devices/cameras/opencv.py \
    --images-dir outputs/images

### Replay recordings
python lerobot/scripts/control_robot.py replay \
    --robot-path lerobot/configs/robot/so100.yaml \
    --fps 30 \
    --repo-id ${HF_USER}/so100_test \
    --episode 0

### Record dataset
python lerobot/scripts/control_robot.py record \
    --robot-path lerobot/configs/robot/so100.yaml \
    --fps 30 \
    --repo-id zhenxuan/lerobot_so100_pick_paper_box_v0 \
    --tags lerobot so100 test \
    --warmup-time-s 5 \
    --episode-time-s 40 \
    --reset-time-s 10 \
    --num-episodes 20 \
    --push-to-hub 1

### Train policy
python lerobot/scripts/train.py \
  dataset_repo_id=zhenxuan/lerobot_so100_pick_paper_box_v0 \
  policy=act_so100_real \
  env=so100_real \
  hydra.run.dir=outputs/train/act_so100_test_20241214 \
  hydra.job.name=act_so100_test \
  device=mps \
  wandb.enable=false



### PingTi

#### Calibration

**Follower Arm**
python lerobot/scripts/control_robot.py \
  --robot.type=pingti \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_follower"]'

**Leader Arm**
python lerobot/scripts/control_robot.py \
  --robot.type=pingti \
  --robot.cameras='{}' \
  --control.type=calibrate \
  --control.arms='["main_leader"]'

##### Teleoperate
python lerobot/scripts/control_robot.py \
  --robot.type=pingti \
  --robot.cameras='{}' \
  --control.type=teleoperate \
  --control.fps=30 \
