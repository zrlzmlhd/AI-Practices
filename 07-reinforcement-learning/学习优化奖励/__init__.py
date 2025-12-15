"""
Learning to Optimize Rewards: A Comprehensive Module Collection

This package provides production-grade implementations of reward optimization
techniques for reinforcement learning, including:

- Potential-Based Reward Shaping (PBRS)
- Inverse Reinforcement Learning (IRL)
- Curiosity-Driven Exploration
- Hindsight Experience Replay (HER)

Each module follows research-grade standards with comprehensive documentation,
mathematical foundations, and extensive testing.

References:
    Reward Shaping:
        [1] Ng et al. (1999). Policy invariance under reward transformations.

    Inverse RL:
        [2] Ng & Russell (2000). Algorithms for inverse reinforcement learning.
        [3] Ziebart et al. (2008). Maximum entropy inverse reinforcement learning.
        [4] Ho & Ermon (2016). Generative adversarial imitation learning.

    Curiosity:
        [5] Pathak et al. (2017). Curiosity-driven exploration by self-supervised prediction.
        [6] Burda et al. (2018). Exploration by random network distillation.

    HER:
        [7] Andrychowicz et al. (2017). Hindsight experience replay.
"""

from .reward_shaping import (
    ShapedRewardConfig,
    RewardShaper,
    DistanceBasedShaper,
    SubgoalBasedShaper,
    LearnedPotentialShaper,
    AdaptiveRewardShaper,
    DynamicShapingConfig,
    compute_optimal_potential_from_value,
    verify_policy_invariance,
)

from .inverse_rl import (
    IRLConfig,
    Demonstration,
    LinearFeatureExtractor,
    InverseRLBase,
    MaxMarginIRL,
    MaxEntropyIRL,
    DeepIRL,
    GAILDiscriminator,
    GAILConfig,
    compute_feature_matching_loss,
    reward_ambiguity_analysis,
)

from .curiosity_driven import (
    CuriosityConfig,
    FeatureEncoder,
    ForwardDynamicsModel,
    InverseDynamicsModel,
    IntrinsicCuriosityModule,
    RandomNetworkDistillation,
    CountBasedExploration,
    EpisodicNoveltyModule,
    compute_exploration_efficiency,
)

from .hindsight_experience_replay import (
    GoalSelectionStrategy,
    Transition,
    Episode,
    HERConfig,
    GoalConditionedReplayBuffer,
    HindsightExperienceReplay,
    PrioritizedHER,
    CurriculumHER,
    GoalGenerator,
    compute_success_rate,
    analyze_goal_coverage,
)

__version__ = "1.0.0"
__author__ = "AI-Practices Contributors"

__all__ = [
    # Reward Shaping
    "ShapedRewardConfig",
    "RewardShaper",
    "DistanceBasedShaper",
    "SubgoalBasedShaper",
    "LearnedPotentialShaper",
    "AdaptiveRewardShaper",
    "DynamicShapingConfig",
    "compute_optimal_potential_from_value",
    "verify_policy_invariance",
    # Inverse RL
    "IRLConfig",
    "Demonstration",
    "LinearFeatureExtractor",
    "InverseRLBase",
    "MaxMarginIRL",
    "MaxEntropyIRL",
    "DeepIRL",
    "GAILDiscriminator",
    "GAILConfig",
    "compute_feature_matching_loss",
    "reward_ambiguity_analysis",
    # Curiosity
    "CuriosityConfig",
    "FeatureEncoder",
    "ForwardDynamicsModel",
    "InverseDynamicsModel",
    "IntrinsicCuriosityModule",
    "RandomNetworkDistillation",
    "CountBasedExploration",
    "EpisodicNoveltyModule",
    "compute_exploration_efficiency",
    # HER
    "GoalSelectionStrategy",
    "Transition",
    "Episode",
    "HERConfig",
    "GoalConditionedReplayBuffer",
    "HindsightExperienceReplay",
    "PrioritizedHER",
    "CurriculumHER",
    "GoalGenerator",
    "compute_success_rate",
    "analyze_goal_coverage",
]
