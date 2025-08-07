#!/usr/bin/env python

"""
Simple demonstration of asynchronous episode saving functionality.

This script shows how to use the async saving feature without requiring
actual robot hardware or teleoperation devices.
"""

import logging
import time
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features


def create_mock_robot_features():
    """Create mock robot features for demonstration."""
    return {
        "action": {"dtype": "float32", "shape": [6]},  # 6-DOF robot arm
        "observation": {"dtype": "float32", "shape": [10]},  # 10-DOF state
    }


def simulate_recording_episode(dataset, episode_length=30):
    """Simulate recording an episode by adding frames."""
    print(f"Recording episode with {episode_length} frames...")
    
    for i in range(episode_length):
        # Simulate robot action and observation
        action = np.random.randn(6).astype(np.float32)
        observation = np.random.randn(10).astype(np.float32)
        
        frame = {
            "action": action,
            "observation": observation,
        }
        
        dataset.add_frame(frame, task="demo_task")
        
        # Simulate recording delay
        time.sleep(0.01)  # 10ms per frame
    
    print(f"Episode recording completed ({episode_length} frames)")


def main():
    """Main demonstration function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("=== Async Episode Saving Demonstration ===\n")
    
    # Create temporary dataset
    temp_dir = Path("/tmp/async_saving_demo")
    temp_dir.mkdir(exist_ok=True)
    
    # Create mock features
    robot_features = create_mock_robot_features()
    action_features = hw_to_dataset_features(robot_features, "action")
    obs_features = hw_to_dataset_features(robot_features, "observation")
    dataset_features = {**action_features, **obs_features}
    
    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id="demo/async_saving_test",
        fps=30,
        features=dataset_features,
        root=temp_dir,
        robot_type="demo_robot",
        use_videos=False,  # Disable videos for faster demo
    )
    
    print("Dataset created successfully")
    
    # Demonstrate synchronous saving (baseline)
    print("\n--- Synchronous Saving (Baseline) ---")
    start_time = time.time()
    
    for episode in range(3):
        episode_start = time.time()
        simulate_recording_episode(dataset, episode_length=20)
        
        # Synchronous save
        save_start = time.time()
        dataset.save_episode()
        save_time = time.time() - save_start
        
        episode_time = time.time() - episode_start
        print(f"Episode {episode + 1}: Recording={episode_time:.2f}s, Save={save_time:.2f}s")
    
    sync_total_time = time.time() - start_time
    print(f"Total time (sync): {sync_total_time:.2f}s")
    
    # Clear dataset for async demo
    dataset = LeRobotDataset.create(
        repo_id="demo/async_saving_test",
        fps=30,
        features=dataset_features,
        root=temp_dir,
        robot_type="demo_robot",
        use_videos=False,
    )
    
    # Enable async saving
    print("\n--- Asynchronous Saving ---")
    dataset.enable_async_saving(max_queue_size=5, save_timeout=60.0)
    print("Async saving enabled")
    
    start_time = time.time()
    
    for episode in range(3):
        episode_start = time.time()
        simulate_recording_episode(dataset, episode_length=20)
        
        # Asynchronous save
        save_start = time.time()
        success = dataset.save_episode_async()
        save_time = time.time() - save_start
        
        episode_time = time.time() - episode_start
        print(f"Episode {episode + 1}: Recording={episode_time:.2f}s, Queue={save_time:.3f}s")
        
        if not success:
            print(f"  Warning: Episode {episode + 1} queue failed, falling back to sync")
            dataset.save_episode()
    
    # Wait for all async saves to complete
    print("\nWaiting for async saves to complete...")
    wait_start = time.time()
    success = dataset.wait_for_async_saves(timeout=30.0)
    wait_time = time.time() - wait_start
    
    if success:
        print(f"All async saves completed in {wait_time:.2f}s")
    else:
        print(f"Timeout waiting for async saves after {wait_time:.2f}s")
    
    async_total_time = time.time() - start_time
    print(f"Total time (async): {async_total_time:.2f}s")
    
    # Compare performance
    print(f"\n--- Performance Comparison ---")
    print(f"Synchronous total time: {sync_total_time:.2f}s")
    print(f"Asynchronous total time: {async_total_time:.2f}s")
    print(f"Time saved: {sync_total_time - async_total_time:.2f}s")
    print(f"Improvement: {((sync_total_time - async_total_time) / sync_total_time * 100):.1f}%")
    
    # Show async save results
    print("\n--- Async Save Results ---")
    results = dataset.get_async_save_results()
    if results:
        successful_saves = sum(1 for r in results if r["success"])
        failed_saves = len(results) - successful_saves
        print(f"Successful saves: {successful_saves}")
        print(f"Failed saves: {failed_saves}")
        
        for result in results:
            status = "✓" if result["success"] else "✗"
            print(f"  {status} Episode {result['episode_index']}: {result['save_time']:.2f}s")
            if not result["success"]:
                print(f"    Error: {result['error_message']}")
    
    # Show final status
    print("\n--- Final Status ---")
    status = dataset.get_async_save_status()
    print(f"Async saving enabled: {status['async_saving_enabled']}")
    print(f"Total submitted: {status['total_submitted']}")
    print(f"Total completed: {status['total_completed']}")
    print(f"Total failed: {status['total_failed']}")
    print(f"Queue size: {status['queue_size']}")
    
    # Cleanup
    print("\n--- Cleanup ---")
    dataset.disable_async_saving(wait_for_completion=True, timeout=10.0)
    print("Async saving disabled")
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("Temporary files cleaned up")
    
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main() 