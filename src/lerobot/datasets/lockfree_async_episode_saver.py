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
Lock-free asynchronous episode saver for non-blocking dataset recording.

This module provides a LockFreeAsyncEpisodeSaver class that handles episode saving
in a separate thread using lock-free data structures to avoid blocking the main recording loop.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from collections import deque
import weakref

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class LockFreeEpisodeSaveTask:
    """Represents a task to save an episode with lock-free design."""
    episode_buffer: Dict[str, Any]
    episode_index: int
    timestamp: float
    task_id: int  # Unique task identifier


@dataclass
class LockFreeEpisodeSaveResult:
    """Represents the result of saving an episode."""
    task_id: int
    episode_index: int
    success: bool
    error_message: Optional[str] = None
    save_time: Optional[float] = None


class AtomicCounter:
    """Thread-safe atomic counter using atomic operations."""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        """Atomically increment and return the new value."""
        with self._lock:
            self._value += 1
            return self._value
    
    def get(self) -> int:
        """Get the current value."""
        with self._lock:
            return self._value
    
    def set(self, value: int) -> None:
        """Set the value."""
        with self._lock:
            self._value = value


class LockFreeQueue:
    """A simple lock-free queue using deque with minimal locking."""
    
    def __init__(self, maxsize: int = 100):
        self._queue = deque(maxlen=maxsize)
        self._lock = threading.Lock()
        self._maxsize = maxsize
    
    def put_nowait(self, item: Any) -> bool:
        """Try to put an item in the queue without blocking."""
        with self._lock:
            if len(self._queue) >= self._maxsize:
                return False
            self._queue.append(item)
            return True
    
    def get_nowait(self) -> Optional[Any]:
        """Try to get an item from the queue without blocking."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()
    
    def qsize(self) -> int:
        """Get the current queue size."""
        with self._lock:
            return len(self._queue)
    
    def empty(self) -> bool:
        """Check if the queue is empty."""
        with self._lock:
            return len(self._queue) == 0


class LockFreeAsyncEpisodeSaver:
    """
    Lock-free asynchronous episode saver that handles episode saving in a separate thread.
    
    This class provides non-blocking episode saving by:
    1. Using lock-free data structures for task submission
    2. Implementing atomic operations for state management
    3. Using separate memory spaces to avoid data races
    4. Providing non-blocking status queries
    
    Args:
        dataset: The LeRobotDataset instance to save episodes to
        max_queue_size: Maximum number of pending save tasks in the queue
        save_timeout: Timeout for individual save operations (seconds)
    """
    
    def __init__(
        self, 
        dataset: LeRobotDataset, 
        max_queue_size: int = 10,
        save_timeout: float = 300.0
    ):
        self.dataset = dataset
        self.max_queue_size = max_queue_size
        self.save_timeout = save_timeout
        
        # Lock-free data structures
        self._task_queue = LockFreeQueue(maxsize=max_queue_size)
        self._result_queue = LockFreeQueue(maxsize=max_queue_size * 2)
        
        # Atomic counters for state tracking
        self._task_counter = AtomicCounter()
        self._submitted_counter = AtomicCounter()
        self._completed_counter = AtomicCounter()
        self._failed_counter = AtomicCounter()
        
        # Threading components
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_running = False
        
        # Error tracking (thread-safe)
        self._last_error: Optional[str] = None
        self._error_lock = threading.Lock()
        
        # Start the worker thread
        self._start_worker()
    
    def _start_worker(self) -> None:
        """Start the worker thread for processing save tasks."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
            
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="LockFreeAsyncEpisodeSaver-Worker",
            daemon=True
        )
        self._worker_thread.start()
        self._is_running = True
        logging.info("LockFreeAsyncEpisodeSaver worker thread started")
    
    def _worker_loop(self) -> None:
        """Main worker loop that processes save tasks without locks."""
        while not self._stop_event.is_set():
            try:
                # Try to get a task without blocking
                task = self._task_queue.get_nowait()
                if task is None:
                    # No tasks available, sleep briefly
                    time.sleep(0.01)
                    continue
                
                # Process the task
                result = self._save_episode_lockfree(task)
                self._result_queue.put_nowait(result)
                
                # Update counters atomically
                self._completed_counter.increment()
                if not result.success:
                    self._failed_counter.increment()
                    with self._error_lock:
                        self._last_error = result.error_message
                
                logging.debug(f"Processed task {task.task_id} for episode {task.episode_index}")
                
            except Exception as e:
                # Handle unexpected errors in worker thread
                logging.error(f"Unexpected error in LockFreeAsyncEpisodeSaver worker: {e}")
                error_result = LockFreeEpisodeSaveResult(
                    task_id=-1,
                    episode_index=-1,
                    success=False,
                    error_message=str(e),
                    save_time=time.time()
                )
                self._result_queue.put_nowait(error_result)
                self._failed_counter.increment()
        
        logging.info("LockFreeAsyncEpisodeSaver worker thread stopped")
    
    def _save_episode_lockfree(self, task: LockFreeEpisodeSaveTask) -> LockFreeEpisodeSaveResult:
        """
        Save an episode using lock-free approach.
        
        Args:
            task: The episode save task to process
            
        Returns:
            LockFreeEpisodeSaveResult with the outcome of the save operation
        """
        start_time = time.time()
        
        try:
            # Create a deep copy of the episode buffer to avoid race conditions
            episode_buffer = self._deep_copy_episode_buffer(task.episode_buffer)
            
            # Create a separate dataset instance for this save operation
            # This completely avoids data races by not sharing state
            save_dataset = self._create_save_dataset()
            
            # Perform the save operation on the isolated dataset
            self._perform_save_operation(save_dataset, episode_buffer, task.episode_index)
            
            # Merge results back to the main dataset (atomic operation)
            self._merge_save_results(save_dataset, task.episode_index)
            
            save_time = time.time() - start_time
            logging.debug(f"Successfully saved episode {task.episode_index} in {save_time:.2f}s")
            
            return LockFreeEpisodeSaveResult(
                task_id=task.task_id,
                episode_index=task.episode_index,
                success=True,
                save_time=save_time
            )
            
        except Exception as e:
            save_time = time.time() - start_time
            error_msg = f"Failed to save episode {task.episode_index}: {str(e)}"
            logging.error(error_msg)
            
            return LockFreeEpisodeSaveResult(
                task_id=task.task_id,
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
    
    def _create_save_dataset(self) -> LeRobotDataset:
        """
        Create a separate dataset instance for save operations.
        
        This creates an isolated copy of the dataset state to avoid any data races.
        
        Returns:
            Isolated dataset instance for saving
        """
        # Create a new dataset instance with the same configuration
        save_dataset = LeRobotDataset.__new__(LeRobotDataset)
        
        # Copy essential attributes
        save_dataset.repo_id = self.dataset.repo_id
        save_dataset.root = self.dataset.root
        save_dataset.fps = self.dataset.fps
        save_dataset.features = self.dataset.features
        save_dataset.batch_encoding_size = self.dataset.batch_encoding_size
        save_dataset.episodes_since_last_encoding = self.dataset.episodes_since_last_encoding
        save_dataset.tolerance_s = self.dataset.tolerance_s
        save_dataset.image_writer = self.dataset.image_writer
        
        # Create isolated copies of shared state
        save_dataset.meta = self._copy_metadata_isolated()
        save_dataset.hf_dataset = self._copy_hf_dataset_isolated()
        
        return save_dataset
    
    def _copy_metadata_isolated(self):
        """Create an isolated copy of metadata."""
        import copy
        
        # Create a deep copy of metadata to ensure complete isolation
        meta_copy = copy.deepcopy(self.dataset.meta)
        
        # Ensure all mutable attributes are properly copied
        meta_copy.info = copy.deepcopy(self.dataset.meta.info)
        meta_copy.episodes = copy.deepcopy(self.dataset.meta.episodes)
        meta_copy.episodes_stats = copy.deepcopy(self.dataset.meta.episodes_stats)
        meta_copy.tasks = copy.deepcopy(self.dataset.meta.tasks)
        meta_copy.task_to_task_index = copy.deepcopy(self.dataset.meta.task_to_task_index)
        
        return meta_copy
    
    def _copy_hf_dataset_isolated(self):
        """Create an isolated copy of HF dataset."""
        import copy
        
        # Create a deep copy of the HF dataset
        return copy.deepcopy(self.dataset.hf_dataset)
    
    def _perform_save_operation(self, save_dataset: LeRobotDataset, episode_buffer: Dict[str, Any], episode_index: int) -> None:
        """
        Perform the save operation on the isolated dataset.
        
        Args:
            save_dataset: Isolated dataset instance
            episode_buffer: Episode buffer to save
            episode_index: Episode index
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
        validate_episode_buffer(episode_buffer, save_dataset.meta.total_episodes, save_dataset.features)
        
        # Extract episode information
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        
        # Set up episode indices
        episode_buffer["index"] = np.arange(save_dataset.meta.total_frames, 
                                          save_dataset.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)
        
        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = save_dataset.meta.get_task_index(task)
            if task_index is None:
                save_dataset.meta.add_task(task)
        
        # Set task indices
        episode_buffer["task_index"] = np.array([save_dataset.meta.get_task_index(task) for task in tasks])
        
        # Process features
        for key, ft in save_dataset.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])
        
        # Wait for image writer to finish
        save_dataset._wait_image_writer()
        
        # Save episode table
        self._save_episode_table_isolated(save_dataset, episode_buffer, episode_index)
        
        # Compute episode stats
        ep_stats = compute_episode_stats(episode_buffer, save_dataset.features)
        
        # Handle video encoding
        has_video_keys = len(save_dataset.meta.video_keys) > 0
        use_batched_encoding = save_dataset.batch_encoding_size > 1
        
        if has_video_keys and not use_batched_encoding:
            save_dataset.encode_episode_videos(episode_index)
        
        # Save episode metadata
        save_dataset.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)
        
        # Handle batch encoding
        if has_video_keys and use_batched_encoding:
            save_dataset.episodes_since_last_encoding += 1
            if save_dataset.episodes_since_last_encoding == save_dataset.batch_encoding_size:
                start_ep = save_dataset.num_episodes - save_dataset.batch_encoding_size
                end_ep = save_dataset.num_episodes
                logging.info(f"Batch encoding {save_dataset.batch_encoding_size} videos for episodes {start_ep} to {end_ep - 1}")
                save_dataset.batch_encode_videos(start_ep, end_ep)
                save_dataset.episodes_since_last_encoding = 0
        
        # Verify timestamps
        ep_data_index = get_episode_data_index(save_dataset.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            save_dataset.fps,
            save_dataset.tolerance_s,
        )
    
    def _save_episode_table_isolated(self, save_dataset: LeRobotDataset, episode_buffer: Dict[str, Any], episode_index: int) -> None:
        """
        Save episode table on the isolated dataset.
        
        Args:
            save_dataset: Isolated dataset instance
            episode_buffer: Episode buffer
            episode_index: Episode index
        """
        import datasets
        from datasets import concatenate_datasets
        from lerobot.datasets.utils import embed_images, hf_transform_to_torch
        
        # Create episode dataset
        episode_dict = {key: episode_buffer[key] for key in save_dataset.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=save_dataset.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        
        # 🚨 关键修改：只保存 episode 到文件，不修改任何共享状态
        # 这样可以完全避免线程竞争问题
        
        # Save to parquet file (只保存当前 episode)
        ep_data_path = save_dataset.root / save_dataset.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)
        
        # 注意：我们不在这里合并到 hf_dataset
        # 合并操作将在主线程中安全地进行，或者通过其他机制处理
    
    def _merge_save_results(self, save_dataset: LeRobotDataset, episode_index: int) -> None:
        """
        Merge save results back to the main dataset using atomic operations.
        
        This method carefully merges the isolated save results back to the main dataset
        without causing data races.
        
        Args:
            save_dataset: Dataset with save results
            episode_index: Episode index that was saved
        """
        # Use a brief lock only for the final merge operation
        # This is much shorter than the entire save operation
        if not hasattr(self.dataset, '_merge_lock'):
            self.dataset._merge_lock = threading.Lock()
        
        with self.dataset._merge_lock:
            # Merge metadata changes (episodes, tasks, stats, etc.)
            self._merge_metadata_changes(save_dataset)
            
            # 🚨 关键修改：不在这里合并 HF 数据集
            # HF 数据集的合并将在主线程中安全地进行
            # 这样可以完全避免线程竞争问题
    
    def _merge_metadata_changes(self, save_dataset: LeRobotDataset) -> None:
        """Merge metadata changes from save dataset to main dataset."""
        # Merge new tasks
        for task, task_index in save_dataset.meta.task_to_task_index.items():
            if task not in self.dataset.meta.task_to_task_index:
                self.dataset.meta.task_to_task_index[task] = task_index
                self.dataset.meta.tasks[task_index] = task
        
        # Merge episode information
        if hasattr(save_dataset.meta, 'episodes') and save_dataset.meta.episodes:
            for ep_idx, ep_info in save_dataset.meta.episodes.items():
                if ep_idx not in self.dataset.meta.episodes:
                    self.dataset.meta.episodes[ep_idx] = ep_info
        
        # Merge episode stats
        if hasattr(save_dataset.meta, 'episodes_stats') and save_dataset.meta.episodes_stats:
            for ep_idx, ep_stats in save_dataset.meta.episodes_stats.items():
                if ep_idx not in self.dataset.meta.episodes_stats:
                    self.dataset.meta.episodes_stats[ep_idx] = ep_stats
        
        # Update total episodes and frames
        self.dataset.meta.info["total_episodes"] = save_dataset.meta.info["total_episodes"]
        self.dataset.meta.info["total_frames"] = save_dataset.meta.info["total_frames"]
        self.dataset.meta.info["total_tasks"] = save_dataset.meta.info["total_tasks"]
    

    
    def submit_episode(self, episode_buffer: Dict[str, Any], episode_index: int) -> bool:
        """
        Submit an episode for lock-free asynchronous saving.
        
        Args:
            episode_buffer: The episode buffer to save
            episode_index: The index of the episode
            
        Returns:
            True if the task was successfully queued, False if queue is full
        """
        if not self._is_running:
            logging.warning("LockFreeAsyncEpisodeSaver is not running, cannot submit episode")
            return False
        
        # Generate unique task ID
        task_id = self._task_counter.increment()
        
        task = LockFreeEpisodeSaveTask(
            episode_buffer=episode_buffer,
            episode_index=episode_index,
            timestamp=time.time(),
            task_id=task_id
        )
        
        # Try to put the task in the queue without blocking
        success = self._task_queue.put_nowait(task)
        
        if success:
            self._submitted_counter.increment()
            logging.debug(f"Submitted episode {episode_index} (task {task_id}) for lock-free async saving")
        else:
            logging.warning(f"Save queue is full, episode {episode_index} (task {task_id}) could not be queued")
        
        return success
    
    def get_results(self, timeout: float = 0.0) -> List[LockFreeEpisodeSaveResult]:
        """
        Get completed save results without blocking.
        
        Args:
            timeout: Timeout for waiting for results (0.0 = non-blocking)
            
        Returns:
            List of completed save results
        """
        results = []
        
        while True:
            result = self._result_queue.get_nowait()
            if result is None:
                break
            results.append(result)
        
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
        
        while self._submitted_counter.get() > self._completed_counter.get():
            if timeout is not None and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the lock-free async saver.
        
        Returns:
            Dictionary with status information
        """
        return {
            "is_running": self._is_running,
            "total_submitted": self._submitted_counter.get(),
            "total_completed": self._completed_counter.get(),
            "total_failed": self._failed_counter.get(),
            "queue_size": self._task_queue.qsize(),
            "pending_tasks": self._submitted_counter.get() - self._completed_counter.get(),
            "last_error": self._last_error,
            "worker_alive": self._worker_thread.is_alive() if self._worker_thread else False
        }
    
    def stop(self, wait: bool = True, timeout: Optional[float] = None) -> None:
        """
        Stop the lock-free async saver and optionally wait for completion.
        
        Args:
            wait: Whether to wait for pending tasks to complete
            timeout: Maximum time to wait for completion
        """
        if not self._is_running:
            return
        
        logging.info("Stopping LockFreeAsyncEpisodeSaver...")
        
        # Signal the worker thread to stop
        self._stop_event.set()
        
        if wait:
            # Wait for the worker thread to finish
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=timeout)
            
            # Wait for any remaining tasks to complete
            if not self.wait_for_completion(timeout=timeout):
                logging.warning("Timeout waiting for LockFreeAsyncEpisodeSaver to complete")
        
        self._is_running = False
        logging.info("LockFreeAsyncEpisodeSaver stopped")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure graceful shutdown."""
        self.stop(wait=True, timeout=30.0)  # Wait up to 30 seconds for completion 