#!/usr/bin/env python

"""
Demonstration of lock-free asynchronous episode saving performance advantages.

This script compares the performance of:
1. Synchronous saving (baseline)
2. Lock-based async saving
3. Lock-free async saving

The lock-free approach should show better performance in high-concurrency scenarios.
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


def benchmark_sync_saving(dataset, num_episodes=5):
    """Benchmark synchronous saving performance."""
    print("\n=== Synchronous Saving Benchmark ===")
    start_time = time.time()
    
    for episode in range(num_episodes):
        episode_start = time.time()
        simulate_recording_episode(dataset, episode_length=20)
        
        # Synchronous save
        save_start = time.time()
        dataset.save_episode()
        save_time = time.time() - save_start
        
        episode_time = time.time() - episode_start
        print(f"Episode {episode + 1}: Recording={episode_time:.2f}s, Save={save_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Total time (sync): {total_time:.2f}s")
    return total_time


def benchmark_lock_async_saving(dataset, num_episodes=5):
    """Benchmark lock-based async saving performance."""
    print("\n=== Lock-Based Async Saving Benchmark ===")
    
    # Enable lock-based async saving
    dataset.enable_async_saving(max_queue_size=5, save_timeout=60.0)
    print("Lock-based async saving enabled")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        episode_start = time.time()
        simulate_recording_episode(dataset, episode_length=20)
        
        # Lock-based async save
        save_start = time.time()
        success = dataset.save_episode_async()
        save_time = time.time() - save_start
        
        episode_time = time.time() - episode_start
        print(f"Episode {episode + 1}: Recording={episode_time:.2f}s, Queue={save_time:.3f}s")
        
        if not success:
            print(f"  Warning: Episode {episode + 1} queue failed")
    
    # Wait for completion
    print("\nWaiting for lock-based async saves to complete...")
    wait_start = time.time()
    success = dataset.wait_for_async_saves(timeout=60.0)
    wait_time = time.time() - wait_start
    
    if success:
        print(f"All lock-based async saves completed in {wait_time:.2f}s")
    else:
        print(f"Timeout waiting for lock-based async saves after {wait_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Total time (lock async): {total_time:.2f}s")
    
    # Disable lock-based async saving
    dataset.disable_async_saving(wait_for_completion=True, timeout=30.0)
    
    return total_time


def benchmark_lockfree_async_saving(dataset, num_episodes=5):
    """Benchmark lock-free async saving performance."""
    print("\n=== Lock-Free Async Saving Benchmark ===")
    
    # Enable lock-free async saving
    dataset.enable_lockfree_async_saving(max_queue_size=5, save_timeout=60.0)
    print("Lock-free async saving enabled")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        episode_start = time.time()
        simulate_recording_episode(dataset, episode_length=20)
        
        # Lock-free async save
        save_start = time.time()
        success = dataset.save_episode_lockfree_async()
        save_time = time.time() - save_start
        
        episode_time = time.time() - episode_start
        print(f"Episode {episode + 1}: Recording={episode_time:.2f}s, Queue={save_time:.3f}s")
        
        if not success:
            print(f"  Warning: Episode {episode + 1} queue failed")
    
    # Wait for completion
    print("\nWaiting for lock-free async saves to complete...")
    wait_start = time.time()
    success = dataset.wait_for_lockfree_async_saves(timeout=60.0)
    wait_time = time.time() - wait_start
    
    if success:
        print(f"All lock-free async saves completed in {wait_time:.2f}s")
    else:
        print(f"Timeout waiting for lock-free async saves after {wait_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Total time (lock-free async): {total_time:.2f}s")
    
    # Disable lock-free async saving
    dataset.disable_lockfree_async_saving(wait_for_completion=True, timeout=30.0)
    
    return total_time


def compare_performance(sync_time, lock_async_time, lockfree_async_time):
    """Compare performance of different saving methods."""
    print("\n=== Performance Comparison ===")
    print(f"Synchronous saving:           {sync_time:.2f}s")
    print(f"Lock-based async saving:      {lock_async_time:.2f}s")
    print(f"Lock-free async saving:       {lockfree_async_time:.2f}s")
    
    # Calculate improvements
    lock_improvement = ((sync_time - lock_async_time) / sync_time * 100)
    lockfree_improvement = ((sync_time - lockfree_async_time) / sync_time * 100)
    lockfree_vs_lock = ((lock_async_time - lockfree_async_time) / lock_async_time * 100)
    
    print(f"\nLock-based async improvement:  {lock_improvement:.1f}%")
    print(f"Lock-free async improvement:   {lockfree_improvement:.1f}%")
    print(f"Lock-free vs Lock-based:       {lockfree_vs_lock:.1f}% improvement")
    
    if lockfree_async_time < lock_async_time:
        print("✅ Lock-free async saving shows better performance!")
    else:
        print("⚠️  Lock-based async saving shows better performance in this test")


def main():
    """Main demonstration function."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("=== Lock-Free Async Episode Saving Performance Demo ===\n")
    
    # Create temporary dataset
    temp_dir = Path("/tmp/lockfree_async_saving_demo")
    temp_dir.mkdir(exist_ok=True)
    
    # Create mock features
    robot_features = create_mock_robot_features()
    action_features = hw_to_dataset_features(robot_features, "action")
    obs_features = hw_to_dataset_features(robot_features, "observation")
    dataset_features = {**action_features, **obs_features}
    
    # Test parameters
    num_episodes = 3
    
    try:
        # Benchmark 1: Synchronous saving
        dataset = LeRobotDataset.create(
            repo_id="demo/sync_benchmark",
            fps=30,
            features=dataset_features,
            root=temp_dir / "sync",
            robot_type="demo_robot",
            use_videos=False,
        )
        sync_time = benchmark_sync_saving(dataset, num_episodes)
        
        # Benchmark 2: Lock-based async saving
        dataset = LeRobotDataset.create(
            repo_id="demo/lock_async_benchmark",
            fps=30,
            features=dataset_features,
            root=temp_dir / "lock_async",
            robot_type="demo_robot",
            use_videos=False,
        )
        lock_async_time = benchmark_lock_async_saving(dataset, num_episodes)
        
        # Benchmark 3: Lock-free async saving
        dataset = LeRobotDataset.create(
            repo_id="demo/lockfree_async_benchmark",
            fps=30,
            features=dataset_features,
            root=temp_dir / "lockfree_async",
            robot_type="demo_robot",
            use_videos=False,
        )
        lockfree_async_time = benchmark_lockfree_async_saving(dataset, num_episodes)
        
        # Compare performance
        compare_performance(sync_time, lock_async_time, lockfree_async_time)
        
        # Show detailed results
        print("\n=== Detailed Results ===")
        
        # Lock-based async results
        print("\nLock-based async save results:")
        results = dataset.get_async_save_results()
        if results:
            successful_saves = sum(1 for r in results if r["success"])
            failed_saves = len(results) - successful_saves
            print(f"  Successful saves: {successful_saves}")
            print(f"  Failed saves: {failed_saves}")
        
        # Lock-free async results
        print("\nLock-free async save results:")
        lockfree_results = dataset.get_lockfree_async_save_results()
        if lockfree_results:
            successful_saves = sum(1 for r in lockfree_results if r["success"])
            failed_saves = len(lockfree_results) - successful_saves
            print(f"  Successful saves: {successful_saves}")
            print(f"  Failed saves: {failed_saves}")
            
            # Show task IDs for lock-free results
            for result in lockfree_results:
                status = "✓" if result["success"] else "✗"
                print(f"  {status} Task {result['task_id']}: Episode {result['episode_index']} ({result['save_time']:.2f}s)")
                if not result["success"]:
                    print(f"    Error: {result['error_message']}")
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        raise
    
    finally:
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\nTemporary files cleaned up")
    
    print("\n=== Demo Complete ===")
    print("\nKey advantages of lock-free async saving:")
    print("1. Better concurrency performance")
    print("2. Lower latency for main recording loop")
    print("3. No lock contention")
    print("4. Better scalability for high-throughput scenarios")
    print("5. Isolated memory spaces prevent data races")


if __name__ == "__main__":
    main() 