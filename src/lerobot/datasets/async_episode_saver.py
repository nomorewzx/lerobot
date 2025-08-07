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
Asynchronous episode saver for non-blocking dataset recording.

This module provides an AsyncEpisodeSaver class that handles episode saving
in a separate thread to avoid blocking the main recording loop.
"""

import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Dict, Optional

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class EpisodeSaveTask:
    """Represents a task to save an episode."""
    episode_buffer: Dict[str, Any]
    episode_index: int
    timestamp: float


@dataclass
class EpisodeSaveResult:
    """Represents the result of saving an episode."""
    episode_index: int
    success: bool
    error_message: Optional[str] = None
    save_time: Optional[float] = None


class AsyncEpisodeSaver:
    """
    Asynchronous episode saver that handles episode saving in a separate thread.
    
    This class provides non-blocking episode saving by:
    1. Accepting episode save requests from the main thread
    2. Processing them in a dedicated worker thread
    3. Providing status updates and error handling
    4. Ensuring graceful shutdown
    
    Args:
        dataset: The LeRobotDataset instance to save episodes to
        max_queue_size: Maximum number of pending save tasks in the queue
        save_timeout: Timeout for individual save operations (seconds)
    """
    
    def __init__(
        self, 
        dataset: LeRobotDataset, 
        max_queue_size: int = 10,
        save_timeout: float = 300.0  # 5 minutes timeout
    ):
        self.dataset = dataset
        self.max_queue_size = max_queue_size
        self.save_timeout = save_timeout
        
        # Threading components
        self._task_queue: Queue[EpisodeSaveTask] = Queue(maxsize=max_queue_size)
        self._result_queue: Queue[EpisodeSaveResult] = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # State tracking
        self._is_running = False
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
        self._last_error: Optional[str] = None
        
        # Start the worker thread
        self._start_worker()
    
    def _start_worker(self) -> None:
        """Start the worker thread for processing save tasks."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
            
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AsyncEpisodeSaver-Worker",
            daemon=True
        )
        self._worker_thread.start()
        self._is_running = True
        logging.info("AsyncEpisodeSaver worker thread started")
    
    def _worker_loop(self) -> None:
        """Main worker loop that processes save tasks."""
        while not self._stop_event.is_set():
            try:
                # Wait for a task with timeout to allow checking stop event
                task = self._task_queue.get(timeout=1.0)
                
                # Process the task
                result = self._save_episode(task)
                self._result_queue.put(result)
                
                # Update counters
                with self._lock:
                    self._total_completed += 1
                    if not result.success:
                        self._total_failed += 1
                        self._last_error = result.error_message
                
                # Mark task as done
                self._task_queue.task_done()
                
            except Empty:
                # No tasks available, continue loop
                continue
            except Exception as e:
                # Handle unexpected errors in worker thread
                logging.error(f"Unexpected error in AsyncEpisodeSaver worker: {e}")
                error_result = EpisodeSaveResult(
                    episode_index=-1,
                    success=False,
                    error_message=str(e),
                    save_time=time.time()
                )
                self._result_queue.put(error_result)
        
        logging.info("AsyncEpisodeSaver worker thread stopped")
    
    def _save_episode(self, task: EpisodeSaveTask) -> EpisodeSaveResult:
        """
        Save an episode with timeout and error handling.
        
        Args:
            task: The episode save task to process
            
        Returns:
            EpisodeSaveResult with the outcome of the save operation
        """
        start_time = time.time()
        
        try:
            # Create a deep copy of the episode buffer to avoid race conditions
            episode_buffer = self._deep_copy_episode_buffer(task.episode_buffer)
            
            # Use a lock to ensure thread-safe access to shared dataset state
            with self._get_save_lock():
                # Call the thread-safe save method
                self._save_episode_thread_safe(episode_buffer, task.episode_index)
            
            save_time = time.time() - start_time
            logging.debug(f"Successfully saved episode {task.episode_index} in {save_time:.2f}s")
            
            return EpisodeSaveResult(
                episode_index=task.episode_index,
                success=True,
                save_time=save_time
            )
            
        except Exception as e:
            save_time = time.time() - start_time
            error_msg = f"Failed to save episode {task.episode_index}: {str(e)}"
            logging.error(error_msg)
            
            return EpisodeSaveResult(
                episode_index=task.episode_index,
                success=False,
                error_message=error_msg,
                save_time=save_time
            )
    
    def _deep_copy_episode_buffer(self, episode_buffer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a deep copy of the episode buffer to avoid race conditions.
        
        Args:
            episode_buffer: The episode buffer to copy
            
        Returns:
            Deep copy of the episode buffer
        """
        import copy
        
        # Create a deep copy to ensure no shared references
        copied_buffer = copy.deepcopy(episode_buffer)
        
        # Ensure numpy arrays are properly copied
        for key, value in copied_buffer.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    copied_buffer[key] = [arr.copy() for arr in value]
        
        return copied_buffer
    
    def _get_save_lock(self):
        """
        Get or create a lock for thread-safe saving.
        
        Returns:
            Threading lock for save operations
        """
        if not hasattr(self.dataset, '_save_lock'):
            self.dataset._save_lock = threading.Lock()
        return self.dataset._save_lock
    
    def _save_episode_thread_safe(self, episode_buffer: Dict[str, Any], episode_index: int) -> None:
        """
        Thread-safe episode saving that avoids data races.
        
        This method performs the same operations as the original save_episode
        but in a thread-safe manner, avoiding conflicts with the main recording thread.
        
        Args:
            episode_buffer: The episode buffer to save
            episode_index: The episode index
        """
        from lerobot.datasets.utils import (
            validate_episode_buffer,
            compute_episode_stats,
            get_episode_data_index,
            check_timestamps_sync,
        )
        import datasets
        from datasets import concatenate_datasets
        from lerobot.datasets.utils import embed_images, hf_transform_to_torch
        
        # Validate the episode buffer
        validate_episode_buffer(episode_buffer, self.dataset.meta.total_episodes, self.dataset.features)
        
        # Extract episode information
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        
        # Set up episode indices
        episode_buffer["index"] = np.arange(self.dataset.meta.total_frames, 
                                          self.dataset.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)
        
        # Add new tasks to the tasks dictionary (thread-safe)
        for task in episode_tasks:
            task_index = self.dataset.meta.get_task_index(task)
            if task_index is None:
                self.dataset.meta.add_task(task)
        
        # Set task indices
        episode_buffer["task_index"] = np.array([self.dataset.meta.get_task_index(task) for task in tasks])
        
        # Process features
        for key, ft in self.dataset.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])
        
        # Wait for image writer to finish
        self.dataset._wait_image_writer()
        
        # Save episode table (thread-safe)
        self._save_episode_table_thread_safe(episode_buffer, episode_index)
        
        # Compute episode stats
        ep_stats = compute_episode_stats(episode_buffer, self.dataset.features)
        
        # Handle video encoding
        has_video_keys = len(self.dataset.meta.video_keys) > 0
        use_batched_encoding = self.dataset.batch_encoding_size > 1
        
        if has_video_keys and not use_batched_encoding:
            self.dataset.encode_episode_videos(episode_index)
        
        # Save episode metadata (thread-safe)
        self.dataset.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)
        
        # Handle batch encoding
        if has_video_keys and use_batched_encoding:
            self.dataset.episodes_since_last_encoding += 1
            if self.dataset.episodes_since_last_encoding == self.dataset.batch_encoding_size:
                start_ep = self.dataset.num_episodes - self.dataset.batch_encoding_size
                end_ep = self.dataset.num_episodes
                logging.info(f"Batch encoding {self.dataset.batch_encoding_size} videos for episodes {start_ep} to {end_ep - 1}")
                self.dataset.batch_encode_videos(start_ep, end_ep)
                self.dataset.episodes_since_last_encoding = 0
        
        # Verify timestamps
        ep_data_index = get_episode_data_index(self.dataset.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.dataset.fps,
            self.dataset.tolerance_s,
        )
    
    def _save_episode_table_thread_safe(self, episode_buffer: Dict[str, Any], episode_index: int) -> None:
        """
        Thread-safe episode table saving.
        
        Args:
            episode_buffer: The episode buffer
            episode_index: The episode index
        """
        import datasets
        from datasets import concatenate_datasets
        from lerobot.datasets.utils import embed_images, hf_transform_to_torch
        
        # Create episode dataset
        episode_dict = {key: episode_buffer[key] for key in self.dataset.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.dataset.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        
        # Concatenate with existing dataset (thread-safe)
        self.dataset.hf_dataset = concatenate_datasets([self.dataset.hf_dataset, ep_dataset])
        self.dataset.hf_dataset.set_transform(hf_transform_to_torch)
        
        # Save to parquet file
        ep_data_path = self.dataset.root / self.dataset.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)
    
    def submit_episode(self, episode_buffer: Dict[str, Any], episode_index: int) -> bool:
        """
        Submit an episode for asynchronous saving.
        
        Args:
            episode_buffer: The episode buffer to save
            episode_index: The index of the episode
            
        Returns:
            True if the task was successfully queued, False if queue is full
        """
        if not self._is_running:
            logging.warning("AsyncEpisodeSaver is not running, cannot submit episode")
            return False
        
        task = EpisodeSaveTask(
            episode_buffer=episode_buffer,
            episode_index=episode_index,
            timestamp=time.time()
        )
        
        try:
            # Try to put the task in the queue without blocking
            self._task_queue.put_nowait(task)
            
            with self._lock:
                self._total_submitted += 1
            
            logging.debug(f"Submitted episode {episode_index} for async saving")
            return True
            
        except Queue.Full:
            logging.warning(f"Save queue is full, episode {episode_index} could not be queued")
            return False
    
    def get_results(self, timeout: float = 0.0) -> list[EpisodeSaveResult]:
        """
        Get completed save results.
        
        Args:
            timeout: Timeout for waiting for results (0.0 = non-blocking)
            
        Returns:
            List of completed save results
        """
        results = []
        
        while True:
            try:
                result = self._result_queue.get_nowait()
                results.append(result)
            except Empty:
                break
        
        return results
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all submitted tasks to complete.
        
        Args:
            timeout: Maximum time to wait (None = wait indefinitely)
            
        Returns:
            True if all tasks completed, False if timeout occurred
        """
        start_time = time.time()
        
        while self._total_submitted > self._total_completed:
            if timeout is not None and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the async saver.
        
        Returns:
            Dictionary with status information
        """
        with self._lock:
            return {
                "is_running": self._is_running,
                "total_submitted": self._total_submitted,
                "total_completed": self._total_completed,
                "total_failed": self._total_failed,
                "queue_size": self._task_queue.qsize(),
                "pending_tasks": self._total_submitted - self._total_completed,
                "last_error": self._last_error,
                "worker_alive": self._worker_thread.is_alive() if self._worker_thread else False
            }
    
    def stop(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        Stop the async saver and optionally wait for completion.
        
        Args:
            wait: Whether to wait for pending tasks to complete
            timeout: Maximum time to wait for completion
        """
        if not self._is_running:
            return
        
        logging.info("Stopping AsyncEpisodeSaver...")
        
        # Signal the worker thread to stop
        self._stop_event.set()
        
        if wait:
            # Wait for the worker thread to finish
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=timeout)
            
            # Wait for any remaining tasks to complete
            if not self.wait_for_completion(timeout=timeout):
                logging.warning("Timeout waiting for AsyncEpisodeSaver to complete")
        
        self._is_running = False
        logging.info("AsyncEpisodeSaver stopped")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure graceful shutdown."""
        self.stop(wait=True, timeout=30.0)  # Wait up to 30 seconds for completion 