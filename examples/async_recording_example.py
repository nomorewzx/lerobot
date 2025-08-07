#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example script demonstrating asynchronous episode saving for non-blocking dataset recording.

This script shows how to use the new async saving feature to prevent the main recording
loop from being blocked by episode saving operations.

Usage:
    python examples/async_recording_example.py

Features demonstrated:
1. Enabling async episode saving
2. Monitoring save status during recording
3. Graceful shutdown with completion waiting
4. Error handling and fallback to sync saving
"""

import logging
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.record import record_loop
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower_end_effector import SO100FollowerEndEffector
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun


def main():
    """Main function demonstrating async recording."""
    
    # Configuration
    NUM_EPISODES = 5
    FPS = 30
    EPISODE_TIME_SEC = 30
    RESET_TIME_SEC = 10
    TASK_DESCRIPTION = "Async recording demonstration task"
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create robot and teleoperator configurations
    # Note: You'll need to adjust these paths for your setup
    robot_config = SO100FollowerConfig(
        port="/dev/tty.usbmodem123456789",  # Adjust to your robot's port
        id="async_demo_robot"
    )
    keyboard_config = KeyboardTeleopConfig()
    
    # Create robot and teleoperator instances
    robot = SO100FollowerEndEffector(robot_config)
    keyboard = KeyboardTeleop(keyboard_config)
    
    # Configure dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}
    
    # Create dataset with async saving enabled
    dataset = LeRobotDataset.create(
        repo_id="your_username/async_recording_demo",  # Adjust to your repo
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )
    
    # Enable async saving with custom configuration
    dataset.enable_async_saving(
        max_queue_size=5,  # Allow up to 5 pending saves
        save_timeout=180.0  # 3 minutes timeout per save
    )
    
    logger.info("Async episode saving enabled")
    
    # Initialize visualization and keyboard listener
    _init_rerun(session_name="async_recording_demo")
    listener, events = init_keyboard_listener()
    
    try:
        # Connect devices
        robot.connect()
        keyboard.connect()
        
        if not robot.is_connected or not keyboard.is_connected:
            raise ValueError("Failed to connect to robot or keyboard")
        
        logger.info("Starting async recording demonstration...")
        
        recorded_episodes = 0
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Recording episode {recorded_episodes + 1} of {NUM_EPISODES}")
            
            # Record episode
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                dataset=dataset,
                teleop=keyboard,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
            )
            
            # Reset environment (skip for last episode)
            if not events["stop_recording"] and recorded_episodes < NUM_EPISODES - 1:
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=keyboard,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
                )
            
            # Handle re-recording
            if events["rerecord_episode"]:
                log_say("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
            
            # Save episode asynchronously
            logger.info(f"Saving episode {recorded_episodes + 1} asynchronously...")
            success = dataset.save_episode_async()
            
            if success:
                logger.info(f"Episode {recorded_episodes + 1} queued for async saving")
            else:
                logger.warning(f"Failed to queue episode {recorded_episodes + 1}, falling back to sync saving")
                dataset.save_episode()
            
            # Monitor async save status
            status = dataset.get_async_save_status()
            logger.info(f"Async save status: {status['pending_tasks']} pending, {status['total_completed']} completed")
            
            recorded_episodes += 1
            
            # Small delay to allow async processing
            time.sleep(0.1)
        
        # Wait for all async saves to complete
        logger.info("Recording completed, waiting for async saves to finish...")
        success = dataset.wait_for_async_saves(timeout=120.0)  # Wait up to 2 minutes
        
        if success:
            logger.info("All async saves completed successfully")
        else:
            logger.warning("Timeout waiting for async saves to complete")
        
        # Get final results
        results = dataset.get_async_save_results()
        if results:
            successful_saves = sum(1 for r in results if r["success"])
            failed_saves = len(results) - successful_saves
            logger.info(f"Final async save results: {successful_saves} successful, {failed_saves} failed")
            
            # Log any errors
            for result in results:
                if not result["success"]:
                    logger.error(f"Episode {result['episode_index']} failed to save: {result['error_message']}")
        
        # Upload to hub (optional)
        # dataset.push_to_hub()
        
    except Exception as e:
        logger.error(f"Error during async recording: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        
        # Disable async saving
        dataset.disable_async_saving(wait_for_completion=True, timeout=30.0)
        
        # Disconnect devices
        robot.disconnect()
        keyboard.disconnect()
        
        if listener is not None:
            listener.stop()
        
        logger.info("Async recording demonstration completed")


if __name__ == "__main__":
    main() 