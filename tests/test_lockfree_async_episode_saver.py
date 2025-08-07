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
Tests for lock-free asynchronous episode saver.
"""

import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lockfree_async_episode_saver import (
    LockFreeAsyncEpisodeSaver,
    LockFreeEpisodeSaveTask,
    AtomicCounter,
    LockFreeQueue,
)
from lerobot.datasets.utils import hw_to_dataset_features


class TestAtomicCounter:
    """Test atomic counter functionality."""
    
    def test_increment(self):
        """Test atomic increment operation."""
        counter = AtomicCounter(initial_value=0)
        
        assert counter.get() == 0
        assert counter.increment() == 1
        assert counter.get() == 1
        assert counter.increment() == 2
        assert counter.get() == 2
    
    def test_set(self):
        """Test atomic set operation."""
        counter = AtomicCounter(initial_value=0)
        
        counter.set(10)
        assert counter.get() == 10
        
        counter.set(5)
        assert counter.get() == 5


class TestLockFreeQueue:
    """Test lock-free queue functionality."""
    
    def test_basic_operations(self):
        """Test basic queue operations."""
        queue = LockFreeQueue(maxsize=3)
        
        # Test empty queue
        assert queue.empty()
        assert queue.qsize() == 0
        assert queue.get_nowait() is None
        
        # Test putting items
        assert queue.put_nowait("item1") is True
        assert queue.put_nowait("item2") is True
        assert queue.put_nowait("item3") is True
        assert queue.qsize() == 3
        
        # Test queue full
        assert queue.put_nowait("item4") is False
        
        # Test getting items
        assert queue.get_nowait() == "item1"
        assert queue.get_nowait() == "item2"
        assert queue.get_nowait() == "item3"
        assert queue.get_nowait() is None
        assert queue.empty()


class TestLockFreeAsyncEpisodeSaver:
    """Test lock-free async episode saver."""
    
    @pytest.fixture
    def temp_dataset(self):
        """Create a temporary dataset for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock features
            robot_features = {
                "action": {"dtype": "float32", "shape": [3]},
                "observation": {"dtype": "float32", "shape": [5]},
            }
            action_features = hw_to_dataset_features(robot_features, "action")
            obs_features = hw_to_dataset_features(robot_features, "observation")
            dataset_features = {**action_features, **obs_features}
            
            dataset = LeRobotDataset.create(
                repo_id="test/lockfree_async",
                fps=30,
                features=dataset_features,
                root=Path(temp_dir),
                robot_type="test_robot",
                use_videos=False,
            )
            
            yield dataset
    
    @pytest.fixture
    def mock_episode_buffer(self):
        """Create a mock episode buffer."""
        return {
            "action": [np.random.randn(3).astype(np.float32) for _ in range(10)],
            "observation": [np.random.randn(5).astype(np.float32) for _ in range(10)],
            "timestamp": [time.time() + i * 0.1 for i in range(10)],
            "episode_index": 0,
            "size": 10,
            "task": ["test_task"] * 10,
        }
    
    def test_initialization(self, temp_dataset):
        """Test saver initialization."""
        saver = LockFreeAsyncEpisodeSaver(
            dataset=temp_dataset,
            max_queue_size=5,
            save_timeout=60.0
        )
        
        assert saver.dataset == temp_dataset
        assert saver.max_queue_size == 5
        assert saver.save_timeout == 60.0
        assert saver._is_running is True
        
        # Clean up
        saver.stop(wait=True, timeout=5.0)
    
    def test_submit_episode(self, temp_dataset, mock_episode_buffer):
        """Test episode submission."""
        saver = LockFreeAsyncEpisodeSaver(
            dataset=temp_dataset,
            max_queue_size=2,
            save_timeout=60.0
        )
        
        # Submit episodes
        success1 = saver.submit_episode(mock_episode_buffer, 0)
        success2 = saver.submit_episode(mock_episode_buffer, 1)
        success3 = saver.submit_episode(mock_episode_buffer, 2)  # Should fail due to queue size
        
        assert success1 is True
        assert success2 is True
        assert success3 is False  # Queue is full
        
        # Check status
        status = saver.get_status()
        assert status["total_submitted"] == 2
        assert status["queue_size"] == 2
        
        # Clean up
        saver.stop(wait=True, timeout=5.0)
    
    def test_episode_saving(self, temp_dataset, mock_episode_buffer):
        """Test actual episode saving."""
        saver = LockFreeAsyncEpisodeSaver(
            dataset=temp_dataset,
            max_queue_size=1,
            save_timeout=60.0
        )
        
        # Submit episode
        success = saver.submit_episode(mock_episode_buffer, 0)
        assert success is True
        
        # Wait for completion
        completion_success = saver.wait_for_completion(timeout=10.0)
        assert completion_success is True
        
        # Get results
        results = saver.get_results()
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].episode_index == 0
        
        # Check status
        status = saver.get_status()
        assert status["total_completed"] == 1
        assert status["total_failed"] == 0
        
        # Clean up
        saver.stop(wait=True, timeout=5.0)
    
    def test_error_handling(self, temp_dataset):
        """Test error handling with invalid episode buffer."""
        saver = LockFreeAsyncEpisodeSaver(
            dataset=temp_dataset,
            max_queue_size=1,
            save_timeout=60.0
        )
        
        # Submit invalid episode buffer
        invalid_buffer = {"invalid": "data"}
        success = saver.submit_episode(invalid_buffer, 0)
        assert success is True
        
        # Wait for completion
        completion_success = saver.wait_for_completion(timeout=10.0)
        assert completion_success is True
        
        # Get results
        results = saver.get_results()
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error_message is not None
        
        # Check status
        status = saver.get_status()
        assert status["total_completed"] == 1
        assert status["total_failed"] == 1
        
        # Clean up
        saver.stop(wait=True, timeout=5.0)
    
    def test_concurrent_submission(self, temp_dataset, mock_episode_buffer):
        """Test concurrent episode submission."""
        import threading
        
        saver = LockFreeAsyncEpisodeSaver(
            dataset=temp_dataset,
            max_queue_size=10,
            save_timeout=60.0
        )
        
        results = []
        
        def submit_episode(episode_index):
            buffer_copy = mock_episode_buffer.copy()
            buffer_copy["episode_index"] = episode_index
            success = saver.submit_episode(buffer_copy, episode_index)
            results.append((episode_index, success))
        
        # Submit episodes concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=submit_episode, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) == 5
        successful_submissions = sum(1 for _, success in results if success)
        assert successful_submissions == 5
        
        # Wait for completion
        completion_success = saver.wait_for_completion(timeout=30.0)
        assert completion_success is True
        
        # Check final status
        status = saver.get_status()
        assert status["total_submitted"] == 5
        assert status["total_completed"] == 5
        
        # Clean up
        saver.stop(wait=True, timeout=5.0)


class TestLeRobotDatasetLockFreeAsync:
    """Test LeRobotDataset integration with lock-free async saving."""
    
    @pytest.fixture
    def temp_dataset(self):
        """Create a temporary dataset for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock features
            robot_features = {
                "action": {"dtype": "float32", "shape": [3]},
                "observation": {"dtype": "float32", "shape": [5]},
            }
            action_features = hw_to_dataset_features(robot_features, "action")
            obs_features = hw_to_dataset_features(robot_features, "observation")
            dataset_features = {**action_features, **obs_features}
            
            dataset = LeRobotDataset.create(
                repo_id="test/dataset_lockfree",
                fps=30,
                features=dataset_features,
                root=Path(temp_dir),
                robot_type="test_robot",
                use_videos=False,
            )
            
            yield dataset
    
    def test_enable_lockfree_async_saving(self, temp_dataset):
        """Test enabling lock-free async saving."""
        # Initially disabled
        status = temp_dataset.get_lockfree_async_save_status()
        assert status["lockfree_async_saving_enabled"] is False
        
        # Enable
        temp_dataset.enable_lockfree_async_saving(max_queue_size=5, save_timeout=60.0)
        
        # Check status
        status = temp_dataset.get_lockfree_async_save_status()
        assert status["lockfree_async_saving_enabled"] is True
        assert status["is_running"] is True
        
        # Disable
        temp_dataset.disable_lockfree_async_saving(wait_for_completion=True, timeout=5.0)
        
        # Check status again
        status = temp_dataset.get_lockfree_async_save_status()
        assert status["lockfree_async_saving_enabled"] is False
    
    def test_save_episode_lockfree_async(self, temp_dataset):
        """Test lock-free async episode saving."""
        # Enable lock-free async saving
        temp_dataset.enable_lockfree_async_saving(max_queue_size=2, save_timeout=60.0)
        
        # Add some frames
        for i in range(5):
            frame = {
                "action": np.random.randn(3).astype(np.float32),
                "observation": np.random.randn(5).astype(np.float32),
            }
            temp_dataset.add_frame(frame, task="test_task")
        
        # Save episode
        success = temp_dataset.save_episode_lockfree_async()
        assert success is True
        
        # Wait for completion
        completion_success = temp_dataset.wait_for_lockfree_async_saves(timeout=10.0)
        assert completion_success is True
        
        # Get results
        results = temp_dataset.get_lockfree_async_save_results()
        assert len(results) == 1
        assert results[0]["success"] is True
        
        # Disable
        temp_dataset.disable_lockfree_async_saving(wait_for_completion=True, timeout=5.0)
    
    def test_fallback_to_sync(self, temp_dataset):
        """Test fallback to synchronous saving when lock-free async is disabled."""
        # Don't enable lock-free async saving
        
        # Add some frames
        for i in range(5):
            frame = {
                "action": np.random.randn(3).astype(np.float32),
                "observation": np.random.randn(5).astype(np.float32),
            }
            temp_dataset.add_frame(frame, task="test_task")
        
        # Save episode (should fallback to sync)
        success = temp_dataset.save_episode_lockfree_async()
        assert success is True
        
        # Check that episode was saved
        assert temp_dataset.num_episodes == 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 