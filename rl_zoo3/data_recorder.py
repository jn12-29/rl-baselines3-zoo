import os
import pickle
import numpy as np
from typing import Any, Dict, List, Optional, Union
from collections import defaultdict
from pathlib import Path


class DataRecorder:
    def __init__(self, output_dir: str, max_episodes: int = 1, prefix: str = "episode"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.max_episodes = max_episodes
        self.prefix = prefix

        self.all_episodes: List[Dict[str, Any]] = []
        self.current_episode_data: Dict[str, Any] = {}

        self._reset_all()

    def _reset_all(self):
        self.current_episode = 0
        self.step_count = 0
        self.episode_returns = np.zeros(self.max_episodes)
        self.current_episode_data = {
            "activations": defaultdict(list),
            "obs": defaultdict(list),
            "actions": [],
            "rewards": [],
            "dones": [],
            "qpos": [],
            "qvel": [],
        }

    def reset_all(self):
        self.all_episodes = []
        self._reset_all()

    def reset_episode(self):
        self.current_episode_data = {
            "activations": defaultdict(list),
            "obs": defaultdict(list),
            "actions": [],
            "rewards": [],
            "dones": [],
            "qpos": [],
            "qvel": [],
        }

    def _get_physics_data(self, env) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            if hasattr(env, "unwrapped"):
                env = env.unwrapped

            if hasattr(env, "data"):
                qpos = np.copy(env.data.qpos) if hasattr(env.data, "qpos") else None
                qvel = np.copy(env.data.qvel) if hasattr(env.data, "qvel") else None
                return qpos, qvel
        except (AttributeError, TypeError):
            pass

        return None, None

    def _prepare_episode_dict(self) -> Dict[str, Any]:
        return {
            "obs": {k: np.array(v) for k, v in self.current_episode_data["obs"].items()},
            "activations": {k: np.array(v) for k, v in self.current_episode_data["activations"].items()},
            "actions": np.array(self.current_episode_data["actions"]),
            "rewards": np.array(self.current_episode_data["rewards"]),
            "dones": np.array(self.current_episode_data["dones"]),
            "qpos": np.array(self.current_episode_data["qpos"]) if self.current_episode_data["qpos"] else None,
            "qvel": np.array(self.current_episode_data["qvel"]) if self.current_episode_data["qvel"] else None,
        }

    def record_step(
        self,
        obs: Union[np.ndarray, Dict],
        action: np.ndarray,
        reward: Union[float, np.ndarray],
        done: Union[bool, np.ndarray],
        env=None,
        activations: Optional[Dict[str, Any]] = None,
        info: Optional[Dict] = None,
    ):
        self.step_count += 1

        if obs is not None:
            if isinstance(obs, dict):
                for k, v in obs.items():
                    self.current_episode_data["obs"][k].append(np.array(v))
            else:
                self.current_episode_data["obs"]["observation"].append(np.array(obs))

        self.current_episode_data["actions"].append(np.array(action))
        self.current_episode_data["rewards"].append(np.array(reward) if isinstance(reward, np.ndarray) else reward)
        self.current_episode_data["dones"].append(np.array(done) if isinstance(done, np.ndarray) else done)

        if activations:
            for layer_name, activation in activations.items():
                for key_name, value in activation.items():
                    self.current_episode_data["activations"][f"{layer_name}_{key_name}"].append(value.cpu().numpy())

        if env is not None:
            qpos, qvel = self._get_physics_data(env)
            if qpos is not None:
                self.current_episode_data["qpos"].append(qpos)
            if qvel is not None:
                self.current_episode_data["qvel"].append(qvel)

        if isinstance(done, np.ndarray):
            for i, d in enumerate(done):
                if d:
                    self.episode_returns[i] += reward[i] if isinstance(reward, np.ndarray) else reward
        else:
            self.episode_returns[0] += reward if isinstance(reward, (int, float)) else np.sum(reward)

        # import pdb

        # pdb.set_trace()

    def end_episode(self) -> Optional[str]:
        if not self.current_episode_data["actions"]:
            return None

        episode_dict = self._prepare_episode_dict()
        episode_dict["episode_length"] = len(episode_dict["actions"])
        episode_dict["episode_return"] = float(np.sum(episode_dict["rewards"]))

        self.all_episodes.append(episode_dict)

        output_file = self.output_dir / f"{self.prefix}_{self.current_episode:04d}.pkl"

        with open(output_file, "wb") as f:
            pickle.dump(episode_dict, f)

        self.current_episode += 1

        self.reset_episode()

        return str(output_file)

    def get_all_episodes(self) -> List[Dict[str, Any]]:
        return [ep.copy() for ep in self.all_episodes]

    def get_episode(self, episode_idx: int) -> Optional[Dict[str, Any]]:
        if 0 <= episode_idx < len(self.all_episodes):
            return self.all_episodes[episode_idx].copy()
        return None

    def get_current_episode(self) -> Dict[str, Any]:
        return self._prepare_episode_dict()

    def _save_metadata(self):
        metadata = {
            "num_episodes": self.current_episode,
            "total_steps": self.step_count,
            "max_episodes": self.max_episodes,
            "output_dir": str(self.output_dir),
            "prefix": self.prefix,
        }

        output_file = self.output_dir / "metadata.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(metadata, f)

    def close(self):
        self.end_episode()
        self._save_metadata()

    def save(self, filepath: str = None) -> str:
        if filepath is None:
            filepath = self.output_dir / "all_data.pkl"

        state = {
            "all_episodes": self.all_episodes,
            "current_episode": self.current_episode,
            "step_count": self.step_count,
            "episode_returns": self.episode_returns,
            "max_episodes": self.max_episodes,
            "output_dir": str(self.output_dir),
            "prefix": self.prefix,
            "current_episode_data": self.current_episode_data,
        }

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        return filepath

    @classmethod
    def load(cls, filepath: str) -> "DataRecorder":
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        output_dir = state["output_dir"]
        recorder = cls(
            output_dir=output_dir,
            max_episodes=state.get("max_episodes", 1),
            prefix=state.get("prefix", "episode"),
        )

        recorder.all_episodes = state.get("all_episodes", [])
        recorder.current_episode = state.get("current_episode", 0)
        recorder.step_count = state.get("step_count", 0)
        recorder.episode_returns = state.get("episode_returns", np.zeros(recorder.max_episodes))
        recorder.current_episode_data = state.get(
            "current_episode_data",
            {
                "activations": defaultdict(list),
                "obs": defaultdict(list),
                "actions": [],
                "rewards": [],
                "dones": [],
                "qpos": [],
                "qvel": [],
            },
        )

        return recorder

    @classmethod
    def load_episode(cls, filepath: str) -> Dict[str, Any]:
        with open(filepath, "rb") as f:
            return pickle.load(f)

    @classmethod
    def load_all_episodes(cls, output_dir: str, prefix: str = "episode") -> List[Dict[str, Any]]:
        dir_path = Path(output_dir)
        episodes = []

        for filepath in sorted(dir_path.glob(f"{prefix}_*.pkl")):
            with open(filepath, "rb") as f:
                episodes.append(pickle.load(f))

        return episodes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
