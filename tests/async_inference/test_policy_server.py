# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Unit-tests for the `PolicyServer` core logic.
Monkey-patch the `policy` attribute with a stub so that no real model inference is performed.
"""

from __future__ import annotations

import logging
import time

import pytest
import torch

from lerobot.configs.types import PolicyFeature
from tests.utils import require_package

# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


class MockPolicy:
    """A minimal mock for an actual policy, returning zeros.
    Refer to tests/policies for tests of the individual policies supported."""

    class _Config:
        robot_type = "dummy_robot"

        @property
        def image_features(self) -> dict[str, PolicyFeature]:
            """Empty image features since this test doesn't use images."""
            return {}

    def predict_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a chunk of 20 dummy actions."""
        batch_size = len(observation["observation.state"])
        return torch.zeros(batch_size, 20, 6)

    def __init__(self):
        self.config = self._Config()

    def to(self, *args, **kwargs):
        # The server calls `policy.to(device)`. This stub ignores it.
        return self

    def model(self, batch: dict) -> torch.Tensor:
        # Return a chunk of 20 dummy actions.
        batch_size = len(batch["robot_type"])
        return torch.zeros(batch_size, 20, 6)


@pytest.fixture
@require_package("grpc")
def policy_server():
    """Fresh `PolicyServer` instance with a stubbed-out policy model."""
    # Import only when the test actually runs (after decorator check)
    from lerobot.scripts.server.configs import PolicyServerConfig
    from lerobot.scripts.server.policy_server import PolicyServer

    test_config = PolicyServerConfig(host="localhost", port=9999)
    server = PolicyServer(test_config)
    # Replace the real policy with our fast, deterministic stub.
    server.policy = MockPolicy()
    server.actions_per_chunk = 20
    server.device = "cpu"

    # Add mock lerobot_features that the observation similarity functions need
    server.lerobot_features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [6],
            "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        }
    }

    return server


# -----------------------------------------------------------------------------
# Helper utilities for tests
# -----------------------------------------------------------------------------


def _make_obs(state: torch.Tensor, timestep: int = 0, must_go: bool = False):
    """Create a TimedObservation with a given state vector."""
    # Import only when needed
    from lerobot.scripts.server.helpers import TimedObservation

    return TimedObservation(
        observation={
            "joint1": state[0].item() if len(state) > 0 else 0.0,
            "joint2": state[1].item() if len(state) > 1 else 0.0,
            "joint3": state[2].item() if len(state) > 2 else 0.0,
            "joint4": state[3].item() if len(state) > 3 else 0.0,
            "joint5": state[4].item() if len(state) > 4 else 0.0,
            "joint6": state[5].item() if len(state) > 5 else 0.0,
        },
        timestamp=time.time(),
        timestep=timestep,
        must_go=must_go,
    )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


def test_time_action_chunk(policy_server):
    """Verify that `_time_action_chunk` assigns correct timestamps and timesteps."""
    start_ts = time.time()
    start_t = 10
    # A chunk of 3 action tensors.
    action_tensors = [torch.randn(6) for _ in range(3)]

    timed_actions = policy_server._time_action_chunk(start_ts, action_tensors, start_t)

    assert len(timed_actions) == 3
    # Check timesteps
    assert [ta.get_timestep() for ta in timed_actions] == [10, 11, 12]
    # Check timestamps
    expected_timestamps = [
        start_ts,
        start_ts + policy_server.config.environment_dt,
        start_ts + 2 * policy_server.config.environment_dt,
    ]
    for ta, expected_ts in zip(timed_actions, expected_timestamps, strict=True):
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


def test_maybe_enqueue_observation_must_go(policy_server):
    """An observation with `must_go=True` is always enqueued."""
    obs = _make_obs(torch.zeros(6), must_go=True)
    assert policy_server._enqueue_observation(obs) is True
    assert policy_server.observation_queue.qsize() == 1
    assert policy_server.observation_queue.get_nowait() is obs


def test_maybe_enqueue_observation_dissimilar(policy_server):
    """A dissimilar observation (not `must_go`) is enqueued."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, dissimilar observation.
    new_obs = _make_obs(torch.ones(6) * 5)  # High norm difference

    assert policy_server._enqueue_observation(new_obs) is True
    assert policy_server.observation_queue.qsize() == 1


def test_maybe_enqueue_observation_is_skipped(policy_server):
    """A similar observation (not `must_go`) is skipped."""
    # Set a last predicted observation.
    policy_server.last_processed_obs = _make_obs(torch.zeros(6))
    # Create a new, very similar observation.
    new_obs = _make_obs(torch.zeros(6) + 1e-4)

    assert policy_server._enqueue_observation(new_obs) is False
    assert policy_server.observation_queue.empty() is True


def test_obs_sanity_checks(policy_server):
    """Unit-test the private `_obs_sanity_checks` helper."""
    prev = _make_obs(torch.zeros(6), timestep=0)

    # Case 1 – timestep already predicted
    policy_server._predicted_timesteps.add(1)
    obs_same_ts = _make_obs(torch.ones(6), timestep=1)
    assert policy_server._obs_sanity_checks(obs_same_ts, prev) is False

    # Case 2 – observation too similar
    policy_server._predicted_timesteps.clear()
    obs_similar = _make_obs(torch.zeros(6) + 1e-4, timestep=2)
    assert policy_server._obs_sanity_checks(obs_similar, prev) is False

    # Case 3 – genuinely new & dissimilar observation passes
    obs_ok = _make_obs(torch.ones(6) * 5, timestep=3)
    assert policy_server._obs_sanity_checks(obs_ok, prev) is True


def test_predict_action_chunk(monkeypatch, policy_server):
    """End-to-end test of `_predict_action_chunk` with a stubbed _get_action_chunk."""
    # Import only when needed
    from lerobot.scripts.server.policy_server import PolicyServer

    # Force server to act-style policy; patch method to return deterministic tensor
    policy_server.policy_type = "act"
    action_dim = 6
    batch_size = 1
    actions_per_chunk = policy_server.actions_per_chunk

    def _fake_get_action_chunk(_self, _obs, _type="act"):
        return torch.zeros(batch_size, actions_per_chunk, action_dim)

    monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)

    obs = _make_obs(torch.zeros(6), timestep=5)
    timed_actions = policy_server._predict_action_chunk(obs)

    assert len(timed_actions) == actions_per_chunk
    assert [ta.get_timestep() for ta in timed_actions] == list(range(5, 5 + actions_per_chunk))

    for i, ta in enumerate(timed_actions):
        expected_ts = obs.get_timestamp() + i * policy_server.config.environment_dt
        assert abs(ta.get_timestamp() - expected_ts) < 1e-6


# -----------------------------------------------------------------------------
# Tests for the serve function
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_grpc_server():
    """Mock gRPC server for testing."""
    from unittest.mock import Mock
    
    mock_server = Mock()
    mock_server.start = Mock()
    mock_server.wait_for_termination = Mock()
    mock_server.add_insecure_port = Mock()
    return mock_server


@pytest.fixture
def mock_thread_pool_executor():
    """Mock ThreadPoolExecutor for testing."""
    from unittest.mock import Mock
    
    mock_executor = Mock()
    return mock_executor


def test_serve_with_valid_config(mock_grpc_server, mock_thread_pool_executor, monkeypatch):
    """Test that serve function works with valid configuration."""
    from unittest.mock import patch, Mock
    from lerobot.scripts.server.configs import PolicyServerConfig
    from lerobot.scripts.server.policy_server import serve
    
    # Mock grpc.server and futures.ThreadPoolExecutor
    with patch('grpc.server', return_value=mock_grpc_server) as mock_grpc_server_func, \
         patch('concurrent.futures.ThreadPoolExecutor', return_value=mock_thread_pool_executor) as mock_executor_func:
        
        # Mock services_pb2_grpc.add_AsyncInferenceServicer_to_server
        mock_add_servicer = Mock()
        with patch('lerobot.scripts.server.policy_server.services_pb2_grpc.add_AsyncInferenceServicer_to_server', 
                  mock_add_servicer):
            
            # Create valid config
            config = PolicyServerConfig(host="localhost", port=8080)
            
            # Call serve function
            serve(config)
            
            # Verify grpc.server was called with ThreadPoolExecutor
            mock_grpc_server_func.assert_called_once()
            mock_executor_func.assert_called_once_with(max_workers=4)
            
            # Verify servicer was added to server
            mock_add_servicer.assert_called_once()
            
            # Verify server was configured with correct port
            mock_grpc_server.add_insecure_port.assert_called_once_with("localhost:8080")
            
            # Verify server was started and waited for termination
            mock_grpc_server.start.assert_called_once()
            mock_grpc_server.wait_for_termination.assert_called_once()


def test_serve_with_custom_config(mock_grpc_server, mock_thread_pool_executor, monkeypatch):
    """Test that serve function works with custom configuration values."""
    from unittest.mock import patch, Mock
    from lerobot.scripts.server.configs import PolicyServerConfig
    from lerobot.scripts.server.policy_server import serve
    
    with patch('grpc.server', return_value=mock_grpc_server), \
         patch('concurrent.futures.ThreadPoolExecutor', return_value=mock_thread_pool_executor), \
         patch('lerobot.scripts.server.policy_server.services_pb2_grpc.add_AsyncInferenceServicer_to_server', Mock()):
        
        # Create config with custom values
        config = PolicyServerConfig(
            host="127.0.0.1", 
            port=9090,
            fps=60,
            inference_latency=0.016,
            obs_queue_timeout=1.5
        )
        
        # Call serve function
        serve(config)
        
        # Verify server was configured with custom port
        mock_grpc_server.add_insecure_port.assert_called_once_with("127.0.0.1:9090")


def test_serve_logs_configuration(mock_grpc_server, mock_thread_pool_executor, monkeypatch, caplog):
    """Test that serve function logs the configuration."""
    from unittest.mock import patch, Mock
    from lerobot.scripts.server.configs import PolicyServerConfig
    from lerobot.scripts.server.policy_server import serve
    
    with patch('grpc.server', return_value=mock_grpc_server), \
         patch('concurrent.futures.ThreadPoolExecutor', return_value=mock_thread_pool_executor), \
         patch('lerobot.scripts.server.policy_server.services_pb2_grpc.add_AsyncInferenceServicer_to_server', Mock()):
        
        config = PolicyServerConfig(host="localhost", port=8080)
        
        with caplog.at_level(logging.INFO):
            serve(config)
        
        # Check that configuration was logged
        assert "host='localhost'" in caplog.text
        assert "port=8080" in caplog.text


def test_serve_creates_policy_server(mock_grpc_server, mock_thread_pool_executor, monkeypatch):
    """Test that serve function creates a PolicyServer instance."""
    from unittest.mock import patch, Mock
    from lerobot.scripts.server.configs import PolicyServerConfig
    from lerobot.scripts.server.policy_server import serve, PolicyServer
    
    with patch('grpc.server', return_value=mock_grpc_server), \
         patch('concurrent.futures.ThreadPoolExecutor', return_value=mock_thread_pool_executor), \
         patch('lerobot.scripts.server.policy_server.services_pb2_grpc.add_AsyncInferenceServicer_to_server') as mock_add_servicer:
        
        config = PolicyServerConfig(host="localhost", port=8080)
        
        serve(config)
        
        # Verify that add_AsyncInferenceServicer_to_server was called
        mock_add_servicer.assert_called_once()
        
        # Get the first argument (the servicer) that was passed to add_AsyncInferenceServicer_to_server
        servicer = mock_add_servicer.call_args[0][0]
        
        # Verify it's a PolicyServer instance
        assert isinstance(servicer, PolicyServer)
        assert servicer.config == config


def test_serve_with_invalid_port_config():
    """Test that serve function raises error with invalid port configuration."""
    from lerobot.scripts.server.configs import PolicyServerConfig
    from lerobot.scripts.server.policy_server import serve
    
    # Test with invalid port (0)
    with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
        config = PolicyServerConfig(host="localhost", port=0)
        serve(config)
    
    # Test with invalid port (65536)
    with pytest.raises(ValueError, match="Port must be between 1 and 65535"):
        config = PolicyServerConfig(host="localhost", port=65536)
        serve(config)


def test_serve_with_invalid_fps_config():
    """Test that serve function raises error with invalid fps configuration."""
    from lerobot.scripts.server.configs import PolicyServerConfig
    from lerobot.scripts.server.policy_server import serve
    
    # Test with invalid fps (0)
    with pytest.raises(ValueError, match="environment_dt must be positive"):
        config = PolicyServerConfig(host="localhost", port=8080, fps=0)
        serve(config)
    
    # Test with invalid fps (negative)
    with pytest.raises(ValueError, match="environment_dt must be positive"):
        config = PolicyServerConfig(host="localhost", port=8080, fps=-1)
        serve(config)


def test_serve_with_invalid_inference_latency_config():
    """Test that serve function raises error with invalid inference_latency configuration."""
    from lerobot.scripts.server.configs import PolicyServerConfig
    from lerobot.scripts.server.policy_server import serve
    
    # Test with invalid inference_latency (negative)
    with pytest.raises(ValueError, match="inference_latency must be non-negative"):
        config = PolicyServerConfig(host="localhost", port=8080, inference_latency=-0.1)
        serve(config)


def test_serve_with_invalid_obs_queue_timeout_config():
    """Test that serve function raises error with invalid obs_queue_timeout configuration."""
    from lerobot.scripts.server.configs import PolicyServerConfig
    from lerobot.scripts.server.policy_server import serve
    
    # Test with invalid obs_queue_timeout (negative)
    with pytest.raises(ValueError, match="obs_queue_timeout must be non-negative"):
        config = PolicyServerConfig(host="localhost", port=8080, obs_queue_timeout=-1.0)
        serve(config)


def test_serve_server_lifecycle(mock_grpc_server, mock_thread_pool_executor, monkeypatch):
    """Test that serve function properly manages server lifecycle."""
    from unittest.mock import patch, Mock
    from lerobot.scripts.server.configs import PolicyServerConfig
    from lerobot.scripts.server.policy_server import serve
    
    with patch('grpc.server', return_value=mock_grpc_server), \
         patch('concurrent.futures.ThreadPoolExecutor', return_value=mock_thread_pool_executor), \
         patch('lerobot.scripts.server.policy_server.services_pb2_grpc.add_AsyncInferenceServicer_to_server', Mock()):
        
        config = PolicyServerConfig(host="localhost", port=8080)
        
        serve(config)
        
        # Verify server lifecycle methods were called in correct order
        mock_grpc_server.add_insecure_port.assert_called_once()
        mock_grpc_server.start.assert_called_once()
        mock_grpc_server.wait_for_termination.assert_called_once()


def test_serve_with_default_config(mock_grpc_server, mock_thread_pool_executor, monkeypatch):
    """Test that serve function works with default configuration."""
    from unittest.mock import patch, Mock
    from lerobot.scripts.server.configs import PolicyServerConfig
    from lerobot.scripts.server.policy_server import serve
    
    with patch('grpc.server', return_value=mock_grpc_server), \
         patch('concurrent.futures.ThreadPoolExecutor', return_value=mock_thread_pool_executor), \
         patch('lerobot.scripts.server.policy_server.services_pb2_grpc.add_AsyncInferenceServicer_to_server', Mock()):
        
        # Create config with defaults
        config = PolicyServerConfig()
        
        serve(config)
        
        # Verify server was configured with default values
        mock_grpc_server.add_insecure_port.assert_called_once_with("localhost:8080")


@patch('grpc.server')
@patch('concurrent.futures.ThreadPoolExecutor')
@patch('lerobot.scripts.server.policy_server.services_pb2_grpc.add_AsyncInferenceServicer_to_server')
def test_serve_thread_pool_workers(mock_add_servicer, mock_executor_class, mock_grpc_server):
    """Test that serve function uses correct number of thread pool workers."""
    from lerobot.scripts.server.configs import PolicyServerConfig
    from lerobot.scripts.server.policy_server import serve
    
    mock_executor = Mock()
    mock_executor_class.return_value = mock_executor
    
    config = PolicyServerConfig(host="localhost", port=8080)
    
    serve(config)
    
    # Verify ThreadPoolExecutor was created with max_workers=4
    mock_executor_class.assert_called_once_with(max_workers=4)
