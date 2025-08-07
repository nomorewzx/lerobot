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
Tests for the AsyncEpisodeSaver functionality.
"""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from lerobot.datasets.async_episode_saver import AsyncEpisodeSaver, EpisodeSaveTask, EpisodeSaveResult
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class TestAsyncEpisodeSaver(unittest.TestCase):
    """Test cases for AsyncEpisodeSaver."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_root = Path(self.temp_dir) / "test_dataset"
        
        # Create a simple dataset for testing
        self.features = {
            "action": {"dtype": "float32", "shape": [3]},
            "observation": {"dtype": "float32", "shape": [5]},
        }
        
        self.dataset = LeRobotDataset.create(
            repo_id="test/async_saver_test",
            fps=30,
            features=self.features,
            root=self.dataset_root,
            robot_type="test_robot",
            use_videos=False,  # Disable videos for faster testing
        )

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'dataset') and hasattr(self.dataset, '_async_saver'):
            self.dataset.disable_async_saving(wait_for_completion=True, timeout=5.0)
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_async_saver_initialization(self):
        """Test AsyncEpisodeSaver initialization."""
        saver = AsyncEpisodeSaver(self.dataset, max_queue_size=5, save_timeout=60.0)
        
        self.assertIsNotNone(saver)
        self.assertEqual(saver.max_queue_size, 5)
        self.assertEqual(saver.save_timeout, 60.0)
        self.assertTrue(saver._is_running)
        
        # Cleanup
        saver.stop(wait=True, timeout=5.0)

    def test_episode_save_task_creation(self):
        """Test EpisodeSaveTask creation."""
        episode_buffer = {
            "action": [np.array([1.0, 2.0, 3.0]) for _ in range(10)],
            "observation": [np.array([1.0, 2.0, 3.0, 4.0, 5.0]) for _ in range(10)],
            "episode_index": 0,
            "size": 10,
            "task": ["test_task"] * 10,
        }
        
        task = EpisodeSaveTask(
            episode_buffer=episode_buffer,
            episode_index=0,
            timestamp=time.time()
        )
        
        self.assertEqual(task.episode_index, 0)
        self.assertEqual(task.episode_buffer["size"], 10)
        self.assertIsInstance(task.timestamp, float)

    def test_episode_save_result_creation(self):
        """Test EpisodeSaveResult creation."""
        result = EpisodeSaveResult(
            episode_index=1,
            success=True,
            save_time=1.5
        )
        
        self.assertEqual(result.episode_index, 1)
        self.assertTrue(result.success)
        self.assertEqual(result.save_time, 1.5)
        self.assertIsNone(result.error_message)

    def test_async_save_basic_functionality(self):
        """Test basic async save functionality."""
        # Enable async saving
        self.dataset.enable_async_saving(max_queue_size=3, save_timeout=30.0)
        
        # Add some frames to the dataset
        for i in range(5):
            frame = {
                "action": np.array([1.0, 2.0, 3.0]),
                "observation": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            }
            self.dataset.add_frame(frame, task="test_task")
        
        # Save episode asynchronously
        success = self.dataset.save_episode_async()
        self.assertTrue(success)
        
        # Wait for completion
        completion_success = self.dataset.wait_for_async_saves(timeout=10.0)
        self.assertTrue(completion_success)
        
        # Check results
        results = self.dataset.get_async_save_results()
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["success"])

    def test_async_save_queue_full(self):
        """Test behavior when save queue is full."""
        # Enable async saving with small queue
        self.dataset.enable_async_saving(max_queue_size=1, save_timeout=30.0)
        
        # Add frames and try to save multiple episodes
        for episode in range(3):
            for i in range(5):
                frame = {
                    "action": np.array([1.0, 2.0, 3.0]),
                    "observation": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                }
                self.dataset.add_frame(frame, task="test_task")
            
            # Try to save
            success = self.dataset.save_episode_async()
            
            if episode == 0:
                # First save should succeed
                self.assertTrue(success)
            else:
                # Subsequent saves should fail due to queue being full
                self.assertFalse(success)

    def test_async_save_status_monitoring(self):
        """Test status monitoring functionality."""
        self.dataset.enable_async_saving(max_queue_size=5, save_timeout=30.0)
        
        # Check initial status
        status = self.dataset.get_async_save_status()
        self.assertTrue(status["async_saving_enabled"])
        self.assertEqual(status["total_submitted"], 0)
        self.assertEqual(status["total_completed"], 0)
        self.assertEqual(status["queue_size"], 0)
        
        # Add frames and save
        for i in range(5):
            frame = {
                "action": np.array([1.0, 2.0, 3.0]),
                "observation": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            }
            self.dataset.add_frame(frame, task="test_task")
        
        success = self.dataset.save_episode_async()
        self.assertTrue(success)
        
        # Check status after submission
        status = self.dataset.get_async_save_status()
        self.assertEqual(status["total_submitted"], 1)
        self.assertEqual(status["queue_size"], 1)
        
        # Wait for completion and check final status
        self.dataset.wait_for_async_saves(timeout=10.0)
        status = self.dataset.get_async_save_status()
        self.assertEqual(status["total_completed"], 1)
        self.assertEqual(status["queue_size"], 0)

    def test_async_save_error_handling(self):
        """Test error handling in async saving."""
        # Create a dataset with invalid features to trigger save errors
        invalid_features = {
            "action": {"dtype": "invalid_type", "shape": [3]},
        }
        
        invalid_dataset = LeRobotDataset.create(
            repo_id="test/invalid_test",
            fps=30,
            features=invalid_features,
            root=self.dataset_root / "invalid",
            robot_type="test_robot",
            use_videos=False,
        )
        
        saver = AsyncEpisodeSaver(invalid_dataset, max_queue_size=2, save_timeout=10.0)
        
        # Try to save an episode (should fail due to invalid features)
        episode_buffer = {
            "action": [np.array([1.0, 2.0, 3.0]) for _ in range(5)],
            "episode_index": 0,
            "size": 5,
            "task": ["test_task"] * 5,
        }
        
        success = saver.submit_episode(episode_buffer, 0)
        self.assertTrue(success)
        
        # Wait for processing
        time.sleep(1.0)
        
        # Check results
        results = saver.get_results()
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIsNotNone(results[0].error_message)
        
        # Cleanup
        saver.stop(wait=True, timeout=5.0)

    def test_async_save_graceful_shutdown(self):
        """Test graceful shutdown of async saver."""
        self.dataset.enable_async_saving(max_queue_size=3, save_timeout=30.0)
        
        # Add frames and submit for saving
        for i in range(5):
            frame = {
                "action": np.array([1.0, 2.0, 3.0]),
                "observation": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            }
            self.dataset.add_frame(frame, task="test_task")
        
        success = self.dataset.save_episode_async()
        self.assertTrue(success)
        
        # Disable async saving with wait
        self.dataset.disable_async_saving(wait_for_completion=True, timeout=10.0)
        
        # Check that async saving is disabled
        status = self.dataset.get_async_save_status()
        self.assertFalse(status["async_saving_enabled"])

    def test_fallback_to_sync_saving(self):
        """Test fallback to synchronous saving when async is disabled."""
        # Don't enable async saving
        self.assertFalse(self.dataset._use_async_saving)
        
        # Add frames
        for i in range(5):
            frame = {
                "action": np.array([1.0, 2.0, 3.0]),
                "observation": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            }
            self.dataset.add_frame(frame, task="test_task")
        
        # Try to save asynchronously (should fallback to sync)
        success = self.dataset.save_episode_async()
        self.assertTrue(success)
        
        # Check that episode was actually saved
        self.assertEqual(self.dataset.num_episodes, 1)


if __name__ == "__main__":
    unittest.main() 