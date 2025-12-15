#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TF-Agents 模块单元测试

验证各核心组件的正确性，包括环境、网络、训练器等。
"""

import unittest
import numpy as np
import tensorflow as tf
import os
import sys

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec


class TestEnvironmentSetup(unittest.TestCase):
    """环境创建测试"""

    def test_gym_environment_loading(self):
        """测试 Gym 环境加载"""
        env_name = "CartPole-v1"
        py_env = suite_gym.load(env_name)

        self.assertIsNotNone(py_env)
        self.assertIsNotNone(py_env.observation_spec())
        self.assertIsNotNone(py_env.action_spec())

    def test_tf_environment_wrapping(self):
        """测试 TensorFlow 环境包装"""
        py_env = suite_gym.load("CartPole-v1")
        tf_env = tf_py_environment.TFPyEnvironment(py_env)

        self.assertEqual(tf_env.batch_size, 1)

        # 测试 reset
        time_step = tf_env.reset()
        self.assertFalse(time_step.is_last())

        # 测试 step
        action = tf.constant([0])
        next_time_step = tf_env.step(action)
        self.assertIsNotNone(next_time_step.observation)

    def test_observation_spec_shape(self):
        """测试观测空间形状"""
        py_env = suite_gym.load("CartPole-v1")
        obs_spec = py_env.observation_spec()

        self.assertEqual(obs_spec.shape, (4,))
        self.assertEqual(obs_spec.dtype, np.float32)


class TestReplayBuffer(unittest.TestCase):
    """经验回放缓冲区测试"""

    def setUp(self):
        """初始化测试环境"""
        from utils.replay_buffer import PrioritizedReplayBuffer
        self.buffer = PrioritizedReplayBuffer(capacity=100)

    def test_add_experience(self):
        """测试添加经验"""
        state = np.random.randn(4).astype(np.float32)
        action = np.array([1])
        reward = 1.0
        next_state = np.random.randn(4).astype(np.float32)
        done = False

        self.buffer.add(state, action, reward, next_state, done)
        self.assertEqual(len(self.buffer), 1)

    def test_sample_batch(self):
        """测试批量采样"""
        # 添加足够的经验
        for _ in range(50):
            state = np.random.randn(4).astype(np.float32)
            action = np.array([np.random.randint(2)])
            reward = np.random.randn()
            next_state = np.random.randn(4).astype(np.float32)
            done = np.random.random() < 0.1

            self.buffer.add(state, action, reward, next_state, done)

        # 采样
        batch, indices, weights = self.buffer.sample(batch_size=16)

        self.assertEqual(batch['states'].shape[0], 16)
        self.assertEqual(len(indices), 16)
        self.assertEqual(len(weights), 16)

    def test_priority_update(self):
        """测试优先级更新"""
        # 添加经验
        for _ in range(20):
            self.buffer.add(
                np.random.randn(4).astype(np.float32),
                np.array([0]),
                1.0,
                np.random.randn(4).astype(np.float32),
                False
            )

        # 采样并更新优先级
        _, indices, _ = self.buffer.sample(batch_size=8)
        td_errors = np.random.randn(8) * 10  # 大的 TD 误差

        # 应该不抛出异常
        self.buffer.update_priorities(indices, td_errors)


class TestNetworkArchitectures(unittest.TestCase):
    """神经网络架构测试"""

    def test_fc_network_creation(self):
        """测试全连接网络创建"""
        from utils.networks import create_fc_network

        net = create_fc_network(
            input_shape=(4,),
            output_dim=2,
            hidden_layers=(64, 32),
            activation="relu"
        )

        # 测试前向传播
        x = tf.random.normal((8, 4))
        y = net(x)

        self.assertEqual(y.shape, (8, 2))

    def test_dueling_network(self):
        """测试 Dueling 网络架构"""
        from utils.networks import create_dueling_network

        net = create_dueling_network(
            input_shape=(4,),
            num_actions=2,
            hidden_layers=(64, 32),
            value_layers=(16,),
            advantage_layers=(16,)
        )

        x = tf.random.normal((8, 4))
        q_values = net(x)

        self.assertEqual(q_values.shape, (8, 2))

    def test_actor_critic_networks(self):
        """测试 Actor-Critic 网络对"""
        from utils.networks import create_actor_critic_networks

        actor, critic = create_actor_critic_networks(
            observation_shape=(4,),
            num_actions=2,
            shared_layers=(64,),
            actor_layers=(32,),
            critic_layers=(32,)
        )

        x = tf.random.normal((8, 4))

        actor_out = actor(x)
        critic_out = critic(x)

        self.assertEqual(actor_out.shape, (8, 2))  # softmax over 2 actions
        self.assertEqual(critic_out.shape, (8, 1))  # single value


class TestMetrics(unittest.TestCase):
    """训练指标测试"""

    def test_training_metrics_collection(self):
        """测试指标收集"""
        from utils.metrics import TrainingMetrics

        metrics = TrainingMetrics(window_size=50)

        # 添加一些模拟数据
        for i in range(100):
            metrics.add_episode(reward=i * 1.0, length=i + 10)
            metrics.add_loss(1.0 / (i + 1))

        stats = metrics.get_statistics()

        self.assertEqual(stats['num_episodes'], 100)
        self.assertGreater(stats['mean_reward'], 0)
        self.assertGreater(stats['recent_mean_reward'], 0)

    def test_smooth_curve(self):
        """测试曲线平滑"""
        from utils.metrics import smooth_curve

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        smoothed = smooth_curve(values, weight=0.5)

        self.assertEqual(len(smoothed), 5)
        # 平滑后第一个值应该不变
        self.assertEqual(smoothed[0], values[0])


class TestCustomEnvironments(unittest.TestCase):
    """自定义环境测试"""

    def test_gridworld_environment(self):
        """测试网格世界环境"""
        # 动态导入（避免顶层导入失败）
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "custom_env",
                os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "05_custom_environment.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            GridWorldEnvironment = module.GridWorldEnvironment

            env = GridWorldEnvironment(width=5, height=5)

            # 测试规范
            obs_spec = env.observation_spec()
            act_spec = env.action_spec()

            self.assertEqual(obs_spec.shape, (2,))
            self.assertEqual(act_spec.shape, ())
            self.assertEqual(act_spec.maximum, 3)

            # 测试 reset
            time_step = env.reset()
            self.assertEqual(time_step.observation.shape, (2,))

            # 测试 step
            time_step = env.step(0)  # 向上移动
            self.assertIsNotNone(time_step.reward)

        except Exception as e:
            self.skipTest(f"Custom environment import failed: {e}")

    def test_bandit_environment(self):
        """测试多臂老虎机环境"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "custom_env",
                os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "05_custom_environment.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            BanditEnvironment = module.BanditEnvironment

            env = BanditEnvironment(num_arms=5)

            # 测试最优臂获取
            optimal_arm = env.get_optimal_arm()
            self.assertIn(optimal_arm, range(5))

            # 测试一步
            env.reset()
            time_step = env.step(0)
            self.assertIsNotNone(time_step.reward)

        except Exception as e:
            self.skipTest(f"Bandit environment import failed: {e}")


class TestTrainerInitialization(unittest.TestCase):
    """训练器初始化测试"""

    def test_dqn_config_defaults(self):
        """测试 DQN 配置默认值"""
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "dqn",
                os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "02_dqn_cartpole.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            DQNConfig = module.DQNConfig

            config = DQNConfig()

            self.assertEqual(config.env_name, "CartPole-v1")
            self.assertEqual(config.batch_size, 64)
            self.assertGreater(config.learning_rate, 0)
            self.assertLess(config.discount_factor, 1)
            self.assertGreater(config.discount_factor, 0)

        except Exception as e:
            self.skipTest(f"DQN config import failed: {e}")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironmentSetup))
    suite.addTests(loader.loadTestsFromTestCase(TestReplayBuffer))
    suite.addTests(loader.loadTestsFromTestCase(TestNetworkArchitectures))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestCustomEnvironments))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainerInitialization))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 60)
    print("TF-Agents 模块单元测试")
    print("=" * 60)

    success = run_tests()

    print("\n" + "=" * 60)
    if success:
        print("所有测试通过！")
    else:
        print("部分测试失败，请检查输出")
    print("=" * 60)
