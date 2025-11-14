# Reinforcement Learning Enhancement Plan for UC-Analyzer

## Vision

Enable the UC-Analyzer to learn from user feedback and expert corrections, continuously improving analysis quality through reinforcement learning.

---

## 1. Core Concept

### What Can Be Learned?

The analyzer makes many **decisions** during analysis that could be learned:

1. **Controller Selection**: Which material controller for a given verb/object context?
2. **Aggregation State Detection**: Solid, liquid, or gas for a material?
3. **Entity Generation**: Which entities to create vs. ignore?
4. **Data Flow Direction**: USE vs. PROVIDE relationships?
5. **Protection Function Triggers**: Which protection functions to activate?
6. **Parallel Group Assignment**: Which steps belong to which parallel group?
7. **Function Assignment**: Which functions belong to which controller?

### Why RL?

- **Delayed Rewards**: Full analysis quality only known after completion
- **Sequential Decisions**: Each decision affects future decisions
- **Exploration**: Try new interpretations, learn from mistakes
- **Adaptation**: Learn domain-specific patterns and user preferences
- **Continuous Improvement**: Get better with every analyzed UC

---

## 2. RL Framework Design

### 2.1 State Representation

**State** = Current context for making a decision

```python
class AnalysisState:
    # Current step being analyzed
    step_id: str              # "B2a"
    step_text: str            # "The system activates the water heater"

    # NLP Analysis
    verb: str                 # "activate"
    verb_lemma: str           # "activate"
    direct_object: str        # "water heater"
    prepositions: List[str]   # []

    # Context
    domain: str               # "beverage_preparation"
    previous_steps: List[str] # ["B1", ...]
    preconditions: List[str]  # ["Water is available", ...]

    # Already generated components
    existing_controllers: List[str]    # ["SystemControlManager", ...]
    existing_entities: List[str]       # ["Water", ...]
    existing_flows: List[Dict]         # [{"from": "A", "to": "B"}, ...]

    # Domain knowledge
    domain_verbs: Dict        # From domain JSON
    domain_materials: Dict    # From domain JSON
    protection_functions: Dict # From domain JSON

    # Encoding (for neural network)
    def to_vector(self) -> np.ndarray:
        """Convert state to fixed-size vector for NN input"""
        pass
```

### 2.2 Action Space

**Actions** = Decisions the analyzer can make

```python
# 1. Controller Selection Action
class ControllerSelectionAction:
    controller_name: str      # "WaterLiquidManager"
    confidence: float         # 0.0-1.0

# 2. Entity Creation Action
class EntityCreationAction:
    entity_name: str          # "HotWater"
    should_create: bool       # True/False
    confidence: float

# 3. Data Flow Action
class DataFlowAction:
    source: str               # "Water"
    target: str               # "WaterLiquidManager"
    flow_type: str            # "USE" or "PROVIDE"
    confidence: float

# 4. Protection Function Action
class ProtectionFunctionAction:
    function_name: str        # "OverheatProtection"
    should_add: bool          # True/False
    confidence: float

# 5. Parallel Group Action
class ParallelGroupAction:
    step_id: str              # "B2a"
    group_number: int         # 0 (serial) or 2, 3, 4, ...
    confidence: float
```

### 2.3 Reward Function

**Reward** = How good was the decision?

```python
class RewardCalculator:
    def calculate_reward(self,
                        action: Action,
                        user_feedback: UserFeedback,
                        validation_result: ValidationResult) -> float:
        """
        Calculate reward from multiple sources:

        1. User Feedback (most important):
           - User accepts: +1.0
           - User corrects: -1.0
           - User deletes: -0.5
           - User rates 5 stars: +2.0
           - User rates 1 star: -2.0

        2. Automatic Validation:
           - Rule violation: -0.5 per violation
           - Rule compliance: +0.2 per rule

        3. Consistency Checks:
           - Orphaned boundary: -0.3
           - Complete flow: +0.1
           - Direct Actor→Controller: -1.0

        4. Time-based:
           - Quick acceptance (<5s): +0.2
           - Slow correction (>60s): -0.1

        5. Expert Review (if available):
           - Expert marks as "good example": +5.0
           - Expert marks as "bad example": -5.0
        """
        reward = 0.0

        # User feedback (primary signal)
        if user_feedback.accepted:
            reward += 1.0
        elif user_feedback.corrected:
            reward -= 1.0
            # Bonus: How far off were we?
            similarity = self._compute_similarity(
                action.result,
                user_feedback.correction
            )
            reward += similarity * 0.5  # Partial credit

        # Validation (secondary signal)
        if validation_result.rules_violated:
            reward -= 0.5 * len(validation_result.rules_violated)
        if validation_result.rules_satisfied:
            reward += 0.2 * len(validation_result.rules_satisfied)

        # Time-based (minor signal)
        if user_feedback.time_to_action < 5.0:
            reward += 0.2  # Quick acceptance = confident good choice

        return reward
```

---

## 3. RL Agent Architectures

### 3.1 Multi-Agent Architecture (Recommended)

**Separate specialized agents for different decision types:**

```
┌─────────────────────────────────────────────────────────────┐
│                     UC Analyzer (Main)                       │
└─────────────────────────────────────────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
   ┌────────────────┐ ┌──────────────┐ ┌─────────────────┐
   │ Controller     │ │ Data Flow    │ │ Protection      │
   │ Selection      │ │ Agent        │ │ Function        │
   │ Agent          │ │              │ │ Agent           │
   └────────────────┘ └──────────────┘ └─────────────────┘
        DQN              Policy Gradient    Multi-Armed Bandit
```

**Advantages**:
- Each agent specializes in one decision type
- Easier to train (smaller state/action spaces)
- Easier to debug
- Can use different RL algorithms for different tasks

### 3.2 Agent 1: Controller Selection Agent

**Task**: Select the best material controller for a UC step

**Algorithm**: Deep Q-Network (DQN)
- **Input**: State vector (step text, verb, object, context)
- **Output**: Q-values for each possible controller
- **Action**: Select controller with highest Q-value

```python
class ControllerSelectionAgent:
    def __init__(self, num_controllers: int):
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_controllers)  # Q-value per controller
        )
        self.target_network = copy.deepcopy(self.q_network)
        self.replay_buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state: AnalysisState, epsilon: float = 0.1) -> str:
        """Epsilon-greedy action selection"""
        if random.random() < epsilon:
            # Explore: Random controller
            return random.choice(self.available_controllers)
        else:
            # Exploit: Best Q-value
            state_vector = state.to_vector()
            q_values = self.q_network(state_vector)
            best_idx = q_values.argmax()
            return self.available_controllers[best_idx]

    def update(self, experience: Experience):
        """Update Q-network from experience"""
        self.replay_buffer.add(experience)

        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(batch.next_states)
            target_q_values = batch.rewards + gamma * next_q_values.max(dim=1)[0]

        # Compute current Q-values
        current_q_values = self.q_network(batch.states)
        current_q_values = current_q_values.gather(1, batch.actions)

        # Loss and backprop
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### 3.3 Agent 2: Protection Function Agent

**Task**: Decide which protection functions to add for a material

**Algorithm**: Multi-Armed Bandit (Contextual Bandit)
- **Input**: Material + Step text context
- **Output**: Probability of adding each protection function
- **Action**: Add protection function if probability > threshold

```python
class ProtectionFunctionAgent:
    """
    Contextual Multi-Armed Bandit for protection function selection.

    Each protection function is an "arm".
    Context = (material, step_text, existing_functions)
    Reward = +1 if appropriate, -1 if not
    """

    def __init__(self, protection_functions: List[str]):
        self.arms = protection_functions
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(protection_functions))  # Score per function
        )

    def select_functions(self, context: Context,
                         threshold: float = 0.5) -> List[str]:
        """Select protection functions to add"""
        context_vector = self._encode_context(context)
        scores = torch.sigmoid(self.context_encoder(context_vector))

        selected = []
        for i, func in enumerate(self.arms):
            if scores[i] > threshold:
                selected.append(func)

        return selected

    def update(self, context: Context,
               selected_functions: List[str],
               rewards: Dict[str, float]):
        """Update based on feedback for each function"""
        context_vector = self._encode_context(context)
        scores = self.context_encoder(context_vector)

        loss = 0.0
        for i, func in enumerate(self.arms):
            if func in selected_functions:
                reward = rewards.get(func, 0.0)
                # Binary cross-entropy for each arm
                target = 1.0 if reward > 0 else 0.0
                loss += F.binary_cross_entropy_with_logits(
                    scores[i],
                    torch.tensor(target)
                )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### 3.4 Agent 3: Data Flow Agent

**Task**: Determine USE vs. PROVIDE relationships

**Algorithm**: Policy Gradient (REINFORCE)
- **Input**: Controller + Entity + Preposition context
- **Output**: Probability distribution over [CREATE_USE, CREATE_PROVIDE, SKIP]
- **Action**: Sample from distribution

```python
class DataFlowAgent:
    def __init__(self):
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # USE, PROVIDE, SKIP
        )
        self.optimizer = optim.Adam(self.policy_network.parameters())
        self.episode_log_probs = []
        self.episode_rewards = []

    def select_action(self, state: FlowState) -> str:
        """Sample action from policy"""
        state_vector = state.to_vector()
        logits = self.policy_network(state_vector)
        probs = F.softmax(logits, dim=-1)

        action_idx = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[action_idx])

        self.episode_log_probs.append(log_prob)

        actions = ["CREATE_USE", "CREATE_PROVIDE", "SKIP"]
        return actions[action_idx]

    def update(self):
        """Update policy at end of episode (full UC analysis)"""
        returns = []
        G = 0
        for r in reversed(self.episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = []
        for log_prob, G in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * G)

        loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear episode memory
        self.episode_log_probs = []
        self.episode_rewards = []
```

---

## 4. Data Collection & Feedback System

### 4.1 Feedback Collection Interface

**User Feedback Mechanisms:**

```python
class FeedbackCollector:
    """
    Collects various forms of user feedback during and after analysis
    """

    def collect_explicit_feedback(self, analysis_id: str):
        """
        Explicit feedback forms:

        1. Star Rating (1-5):
           - Overall analysis quality
           - Per-component ratings

        2. Accept/Reject Buttons:
           - Accept controller selection ✓
           - Reject and correct ✗

        3. Corrections:
           - User changes controller name
           - User adds/removes entity
           - User modifies data flow

        4. Comments:
           - User explains why correction was made
        """
        pass

    def collect_implicit_feedback(self, analysis_id: str):
        """
        Implicit feedback from user actions:

        1. Time Metrics:
           - Time to first correction
           - Total time spent reviewing
           - Time per component type

        2. Edit Actions:
           - Delete component → Strong negative
           - Modify component → Moderate negative
           - Add missing component → Negative (missed it)
           - No edits → Positive (accepted)

        3. Usage Patterns:
           - Immediately use diagram → Positive
           - Regenerate analysis → Negative
           - Export to other tools → Positive
        """
        pass

    def collect_validation_feedback(self, analysis_result: AnalysisResult):
        """
        Automatic validation feedback:

        1. UC-Methode Rule Checks:
           - Rule 1-5 compliance
           - Actor→Boundary→Controller pattern
           - Parallel flow node pairs

        2. Consistency Checks:
           - All boundaries connected
           - No orphaned entities
           - No circular data flows

        3. Completeness Checks:
           - All UC steps analyzed
           - All preconditions addressed
           - All actors handled
        """
        pass
```

### 4.2 Experience Storage

**Store every decision with context and outcome:**

```python
@dataclass
class Experience:
    """Single RL experience (SARS' tuple)"""

    # Context
    uc_id: str
    step_id: str
    domain: str
    timestamp: datetime

    # State
    state: AnalysisState

    # Action
    decision_type: str        # "controller_selection", "data_flow", etc.
    action_taken: Any
    agent_confidence: float

    # Result
    result: Any               # What was generated

    # Feedback (may arrive later)
    user_accepted: Optional[bool] = None
    user_correction: Optional[Any] = None
    validation_score: Optional[float] = None

    # Reward (computed after feedback)
    reward: Optional[float] = None

class ExperienceDatabase:
    """
    Persistent storage for experiences

    SQLite database schema:

    TABLE experiences (
        id INTEGER PRIMARY KEY,
        uc_id TEXT,
        step_id TEXT,
        domain TEXT,
        timestamp DATETIME,
        decision_type TEXT,
        state_vector BLOB,      -- Pickled state
        action BLOB,            -- Pickled action
        result BLOB,            -- Pickled result
        user_accepted BOOLEAN,
        user_correction BLOB,
        validation_score REAL,
        reward REAL,
        INDEX(uc_id),
        INDEX(decision_type),
        INDEX(timestamp)
    );

    TABLE analysis_sessions (
        id INTEGER PRIMARY KEY,
        uc_id TEXT,
        start_time DATETIME,
        end_time DATETIME,
        user_rating INTEGER,    -- 1-5 stars
        user_comment TEXT,
        validation_score REAL,
        total_experiences INTEGER
    );
    """

    def __init__(self, db_path: str = "rl_data/experiences.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def store_experience(self, exp: Experience):
        """Store single experience"""
        pass

    def update_experience_feedback(self, exp_id: int,
                                   feedback: UserFeedback):
        """Update experience with user feedback"""
        pass

    def get_training_batch(self, batch_size: int,
                          decision_type: str = None) -> List[Experience]:
        """Sample batch for training"""
        pass

    def get_statistics(self) -> Dict:
        """Get statistics about collected data"""
        return {
            "total_experiences": self._count_experiences(),
            "experiences_with_feedback": self._count_with_feedback(),
            "avg_reward_by_type": self._avg_reward_by_type(),
            "acceptance_rate": self._compute_acceptance_rate()
        }
```

---

## 5. Training Pipeline

### 5.1 Offline Training (Initial Model)

**Train on historical data before deployment:**

```python
class OfflineTrainer:
    """
    Train RL agents on collected historical data

    Data sources:
    1. Previously analyzed UCs with expert corrections
    2. Synthetic UCs with known correct analyses
    3. Common patterns from domain knowledge
    """

    def prepare_training_data(self):
        """
        Load and preprocess historical data

        Steps:
        1. Load all experiences from database
        2. Filter experiences with feedback
        3. Compute rewards
        4. Split train/validation/test sets
        5. Balance dataset (equal positive/negative samples)
        """
        experiences = self.db.load_all_experiences()

        # Filter: Only keep experiences with feedback
        labeled = [e for e in experiences if e.user_accepted is not None]

        # Compute rewards
        for exp in labeled:
            exp.reward = self.reward_calculator.calculate_reward(exp)

        # Split 80/10/10
        train, val, test = self._split_data(labeled, [0.8, 0.1, 0.1])

        return train, val, test

    def train_agent(self, agent: RLAgent,
                   train_data: List[Experience],
                   val_data: List[Experience],
                   epochs: int = 100):
        """
        Train single agent

        Training loop:
        1. Sample batch from training data
        2. Compute loss
        3. Update agent
        4. Evaluate on validation set
        5. Early stopping if no improvement
        """
        best_val_score = -float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_loss = 0.0
            for batch in self._batch_generator(train_data, batch_size=32):
                loss = agent.update(batch)
                train_loss += loss

            # Validation
            val_score = self._evaluate_agent(agent, val_data)

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
                  f"Val Score = {val_score:.4f}")

            # Early stopping
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                self._save_checkpoint(agent, f"best_agent_epoch{epoch}.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Load best model
        agent.load_checkpoint(f"best_agent_epoch{best_epoch}.pth")

        return agent

    def evaluate_on_test_set(self, agent: RLAgent,
                            test_data: List[Experience]):
        """
        Final evaluation on held-out test set

        Metrics:
        - Accuracy: % of correct decisions
        - Precision/Recall: For each decision type
        - Reward: Average cumulative reward
        - Rule Compliance: % UC-Methode rules satisfied
        """
        metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "avg_reward": 0.0,
            "rule_compliance": 0.0
        }

        for exp in test_data:
            # Agent makes decision
            predicted_action = agent.select_action(exp.state, epsilon=0.0)

            # Compare with ground truth (user correction or acceptance)
            if exp.user_accepted:
                correct = (predicted_action == exp.action_taken)
            else:
                correct = (predicted_action == exp.user_correction)

            if correct:
                metrics["accuracy"] += 1

        metrics["accuracy"] /= len(test_data)

        return metrics
```

### 5.2 Online Learning (During Usage)

**Continuously learn from user interactions:**

```python
class OnlineTrainer:
    """
    Update agents in real-time based on user feedback

    Modes:
    1. Assistant Mode: Agent suggests, user decides
    2. Auto Mode: Agent decides, user can override
    3. Hybrid: Agent decides with confidence threshold
    """

    def __init__(self, agents: Dict[str, RLAgent]):
        self.agents = agents
        self.current_episode = []
        self.update_frequency = 10  # Update after N experiences
        self.experience_counter = 0

    def process_decision_feedback(self,
                                  decision_type: str,
                                  state: AnalysisState,
                                  action: Action,
                                  feedback: UserFeedback):
        """
        Process feedback for a single decision

        Flow:
        1. Store experience
        2. Compute reward
        3. Add to agent's replay buffer
        4. Periodically trigger training update
        """
        # Create experience
        exp = Experience(
            state=state,
            decision_type=decision_type,
            action_taken=action,
            user_accepted=feedback.accepted,
            user_correction=feedback.correction,
            reward=self.compute_reward(action, feedback)
        )

        # Store in database
        self.db.store_experience(exp)

        # Add to current episode
        self.current_episode.append(exp)

        # Get appropriate agent
        agent = self.agents[decision_type]

        # Add to agent's buffer
        agent.replay_buffer.add(exp)

        # Increment counter
        self.experience_counter += 1

        # Periodic update
        if self.experience_counter % self.update_frequency == 0:
            self._trigger_update(agent)

    def _trigger_update(self, agent: RLAgent):
        """
        Trigger training update for agent

        Online learning approaches:
        1. Batch update from replay buffer
        2. Incremental update (single experience)
        3. Mini-batch update (last N experiences)
        """
        if len(agent.replay_buffer) >= agent.min_buffer_size:
            # Sample batch and update
            batch = agent.replay_buffer.sample(batch_size=32)
            loss = agent.update(batch)

            print(f"[ONLINE LEARNING] Updated {agent.name}, Loss = {loss:.4f}")

    def end_analysis_episode(self, overall_feedback: UserFeedback):
        """
        Called when full UC analysis is complete

        For policy gradient agents that need full episode rewards
        """
        # Compute episode return for each experience
        returns = self._compute_returns(self.current_episode)

        # Update policy gradient agents
        for exp, G in zip(self.current_episode, returns):
            agent = self.agents[exp.decision_type]
            if isinstance(agent, PolicyGradientAgent):
                agent.episode_rewards.append(G)

        # Trigger policy updates
        for agent_name, agent in self.agents.items():
            if isinstance(agent, PolicyGradientAgent):
                agent.update()

        # Clear episode buffer
        self.current_episode = []
```

### 5.3 Transfer Learning

**Transfer knowledge between domains:**

```python
class TransferLearner:
    """
    Transfer learned knowledge between domains

    Scenarios:
    1. beverage_preparation → food_preparation
    2. rocket_science → aerospace
    3. Common patterns (all domains)
    """

    def pretrain_on_common_patterns(self, agent: RLAgent):
        """
        Pre-train on domain-independent patterns:
        - Time-based triggers → TimingBoundary
        - User interactions → HMI pattern
        - Error handling → SystemControl
        """
        common_data = self.db.load_common_pattern_data()
        self.offline_trainer.train_agent(agent, common_data)

    def fine_tune_for_domain(self, agent: RLAgent,
                            source_domain: str,
                            target_domain: str):
        """
        Fine-tune agent trained on source domain for target domain

        Approach:
        1. Load agent trained on source domain
        2. Freeze lower layers (feature extraction)
        3. Fine-tune upper layers on target domain data
        """
        # Load source domain agent
        source_agent = torch.load(f"models/{source_domain}_agent.pth")

        # Copy weights to target agent
        agent.load_state_dict(source_agent.state_dict())

        # Freeze lower layers
        for param in agent.feature_extractor.parameters():
            param.requires_grad = False

        # Fine-tune on target domain
        target_data = self.db.load_domain_data(target_domain)
        self.offline_trainer.train_agent(agent, target_data, epochs=20)
```

---

## 6. Integration Architecture

### 6.1 System Architecture with RL

```
┌─────────────────────────────────────────────────────────────────┐
│                        UC Analyzer (Enhanced)                    │
│                                                                  │
│  ┌────────────────┐         ┌──────────────────┐               │
│  │ Traditional    │         │ RL Enhancement   │               │
│  │ Rule-Based     │◄────────┤ Layer            │               │
│  │ Engine         │         │                  │               │
│  └────────────────┘         │ ┌──────────────┐ │               │
│         │                   │ │ Controller   │ │               │
│         │                   │ │ Agent        │ │               │
│         ▼                   │ └──────────────┘ │               │
│  ┌────────────────┐         │ ┌──────────────┐ │               │
│  │ Decision Point │────────►│ │ DataFlow     │ │               │
│  └────────────────┘         │ │ Agent        │ │               │
│         │                   │ └──────────────┘ │               │
│         │                   │ ┌──────────────┐ │               │
│         ▼                   │ │ Protection   │ │               │
│  ┌────────────────┐         │ │ Agent        │ │               │
│  │ Generate RA    │         │ └──────────────┘ │               │
│  │ Components     │         └──────────────────┘               │
│  └────────────────┘                                             │
│         │                                                        │
│         ▼                                                        │
│  ┌────────────────────────────────────────┐                    │
│  │ Feedback Collection & Learning         │                    │
│  │  ┌──────────────┐  ┌──────────────┐   │                    │
│  │  │ User Actions │  │ Validation   │   │                    │
│  │  │ Tracker      │  │ Rules        │   │                    │
│  │  └──────────────┘  └──────────────┘   │                    │
│  │  ┌──────────────┐  ┌──────────────┐   │                    │
│  │  │ Experience   │  │ Online       │   │                    │
│  │  │ Database     │  │ Trainer      │   │                    │
│  │  └──────────────┘  └──────────────┘   │                    │
│  └────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Integration Points in Analyzer

**Modify existing analyzer to support RL:**

```python
# src/rl_integrated_uc_analyzer.py

class RLIntegratedUCAnalyzer(StructuredUCAnalyzer):
    """
    UC Analyzer enhanced with Reinforcement Learning

    Maintains backward compatibility:
    - Can run without RL (traditional mode)
    - Can run with RL suggestions (assistant mode)
    - Can run with RL auto-decisions (auto mode)
    """

    def __init__(self, domain_name: str = DEFAULT_DOMAIN,
                 rl_mode: str = "off"):
        """
        rl_mode options:
        - "off": Traditional rule-based only
        - "assistant": RL suggests, user decides
        - "auto": RL decides, user can override
        - "learning": Auto mode + active learning
        """
        super().__init__(domain_name)

        self.rl_mode = rl_mode

        if rl_mode != "off":
            self.rl_agents = self._load_rl_agents()
            self.feedback_collector = FeedbackCollector()
            self.experience_db = ExperienceDatabase()
            self.online_trainer = OnlineTrainer(self.rl_agents)

    def _generate_controller_for_step_with_rl(self,
                                              step_id: str,
                                              grammatical: GrammaticalAnalysis,
                                              line_text: str) -> Optional[RAClass]:
        """
        Enhanced controller generation with RL support

        Flow:
        1. Traditional rule-based suggests controller
        2. RL agent suggests alternative (if rl_mode != "off")
        3. Compare suggestions:
           - If agree: Use it (high confidence)
           - If disagree:
             * assistant mode: Show both, user decides
             * auto mode: Use RL if confidence > threshold
        4. Collect feedback on chosen controller
        """
        # State for RL agent
        state = AnalysisState(
            step_id=step_id,
            step_text=line_text,
            verb=grammatical.verb_lemma,
            direct_object=grammatical.direct_object,
            domain=self.domain_name,
            # ... context ...
        )

        # Traditional suggestion
        trad_controller = self._generate_controller_using_registry(
            step_id, grammatical, line_text
        )

        if self.rl_mode == "off":
            return trad_controller

        # RL suggestion
        rl_agent = self.rl_agents["controller_selection"]
        rl_suggestion = rl_agent.select_action(state)
        rl_confidence = rl_agent.get_confidence()

        # Decision logic
        if self.rl_mode == "assistant":
            # Show both suggestions to user
            chosen = self._ask_user_to_choose(
                traditional=trad_controller,
                rl_suggestion=rl_suggestion,
                rl_confidence=rl_confidence
            )
        elif self.rl_mode in ["auto", "learning"]:
            # Auto-decide based on confidence
            if rl_confidence > 0.8:
                chosen = rl_suggestion
                print(f"[RL AUTO] Using RL suggestion: {rl_suggestion.name} "
                      f"(confidence: {rl_confidence:.2f})")
            else:
                chosen = trad_controller
                print(f"[RL AUTO] Using traditional: {trad_controller.name} "
                      f"(RL confidence too low: {rl_confidence:.2f})")

        # Collect experience (feedback comes later)
        experience = Experience(
            state=state,
            decision_type="controller_selection",
            action_taken=chosen,
            agent_confidence=rl_confidence
        )
        self.experience_db.store_experience(experience)

        return chosen

    def collect_user_feedback_for_analysis(self, analysis_result: AnalysisResult):
        """
        Collect feedback after analysis is complete

        Called by UI or CLI after user reviews the analysis
        """
        # Overall rating
        rating = self._prompt_user_rating()

        # Component-level feedback
        for component in analysis_result.controllers:
            feedback = self._get_component_feedback(component)
            if feedback.correction:
                # User corrected this component
                exp_id = self._find_experience_id(component)
                self.experience_db.update_experience_feedback(
                    exp_id, feedback
                )

                # Online learning update
                if self.rl_mode == "learning":
                    self.online_trainer.process_decision_feedback(
                        decision_type="controller_selection",
                        state=component.state,
                        action=component,
                        feedback=feedback
                    )
```

---

## 7. Implementation Phases

### Phase 1: Data Collection (1-2 months)

**Goal**: Collect sufficient data for training

**Tasks**:
1. Add logging to analyzer
   - Log every decision with full context
   - Log state, action, result
   - Timestamp everything

2. Implement feedback UI
   - Star rating for overall analysis
   - Accept/Reject buttons per component
   - Correction interface
   - Comment field

3. Automatic validation
   - UC-Methode rule checker
   - Consistency checker
   - Completeness checker

4. Database setup
   - SQLite for experiences
   - Backup strategy
   - Privacy considerations

**Metrics**:
- Target: 1000+ analyzed UCs with feedback
- Target: 10,000+ individual decisions with feedback

### Phase 2: Offline Training (1 month)

**Goal**: Train initial RL agents on collected data

**Tasks**:
1. Prepare training data
   - Clean and preprocess
   - Compute rewards
   - Split train/val/test

2. Implement RL agents
   - Controller Selection Agent (DQN)
   - Protection Function Agent (Contextual Bandit)
   - Data Flow Agent (Policy Gradient)

3. Training pipeline
   - Hyperparameter tuning
   - Cross-validation
   - Early stopping

4. Evaluation
   - Accuracy metrics
   - Comparison with rule-based
   - User study

**Success Criteria**:
- RL agents achieve ≥80% accuracy on test set
- RL agents outperform pure rule-based on at least 2/3 decision types

### Phase 3: Assistant Mode (2 months)

**Goal**: Deploy RL agents as assistants, collect more feedback

**Tasks**:
1. Integrate agents into analyzer
   - RL suggests, user decides
   - Show confidence scores
   - Explain suggestions

2. A/B testing
   - Control group: Traditional only
   - Treatment group: RL assistant
   - Compare quality and speed

3. Continuous learning
   - Online learning from corrections
   - Periodic retraining
   - Model versioning

4. User experience
   - Intuitive UI for suggestions
   - Easy correction workflow
   - Feedback on feedback (meta!)

**Metrics**:
- User acceptance rate of RL suggestions
- Time saved compared to traditional
- User satisfaction scores

### Phase 4: Auto Mode (2 months)

**Goal**: Let RL agents make decisions automatically

**Tasks**:
1. Confidence thresholds
   - Determine safe confidence levels
   - Fallback to traditional for low confidence
   - User override always available

2. Active learning
   - Identify uncertain decisions
   - Ask user for help on those
   - Learn maximally from minimal feedback

3. Multi-domain support
   - Train separate agents per domain
   - Transfer learning between domains
   - Universal agent for common patterns

4. Production deployment
   - Model serving infrastructure
   - Monitoring and alerting
   - Rollback capability

**Success Criteria**:
- RL auto mode achieves same quality as traditional + manual correction
- 50%+ reduction in manual effort
- No degradation in user satisfaction

### Phase 5: Advanced Features (Ongoing)

**Goal**: Continuously improve and expand

**Features**:
1. Curriculum learning
   - Start with easy UCs
   - Progress to complex UCs
   - Learn incrementally

2. Multi-agent coordination
   - Agents cooperate on decisions
   - Shared experience replay
   - Joint training

3. Explainability
   - Why did agent choose X?
   - What features were important?
   - Attention visualization

4. Personalization
   - Learn user preferences
   - Adapt to user's domain expertise
   - Custom agent per user/team

---

## 8. Evaluation Metrics

### 8.1 Agent Performance Metrics

```python
class AgentEvaluator:
    """Comprehensive evaluation of RL agents"""

    def evaluate(self, agent: RLAgent, test_data: List[Experience]) -> Dict:
        return {
            # Accuracy metrics
            "accuracy": self._compute_accuracy(agent, test_data),
            "precision": self._compute_precision(agent, test_data),
            "recall": self._compute_recall(agent, test_data),
            "f1_score": self._compute_f1(agent, test_data),

            # Reward metrics
            "avg_reward": self._compute_avg_reward(agent, test_data),
            "cumulative_reward": self._compute_cumulative_reward(agent, test_data),

            # Confidence metrics
            "avg_confidence": self._compute_avg_confidence(agent, test_data),
            "confidence_calibration": self._compute_calibration(agent, test_data),

            # Efficiency metrics
            "inference_time_ms": self._measure_inference_time(agent),
            "memory_usage_mb": self._measure_memory(agent),

            # Comparison with baseline
            "improvement_over_baseline": self._compare_to_baseline(agent, test_data)
        }
```

### 8.2 System-Level Metrics

**Track overall impact of RL enhancement:**

1. **Quality Metrics**:
   - UC-Methode rule compliance rate
   - Manual correction rate
   - User satisfaction (1-5 stars)
   - Expert review scores

2. **Efficiency Metrics**:
   - Time to complete analysis
   - Number of user corrections needed
   - Time saved vs. traditional approach

3. **Learning Metrics**:
   - Improvement over time
   - Data efficiency (performance vs. training data size)
   - Transfer learning effectiveness

4. **Business Metrics**:
   - User adoption rate
   - Retention rate
   - Productivity gains

---

## 9. Risks & Mitigation

### Risk 1: Insufficient Training Data

**Risk**: Not enough labeled data to train agents effectively

**Mitigation**:
- Start with data augmentation (synthetic UCs)
- Use semi-supervised learning (learn from unlabeled UCs)
- Transfer learning from related domains
- Active learning (ask user for feedback on uncertain cases)

### Risk 2: Distribution Shift

**Risk**: User behavior changes over time, agent performance degrades

**Mitigation**:
- Continuous monitoring of agent performance
- Regular retraining on recent data
- Detect and alert on distribution shift
- Maintain traditional fallback

### Risk 3: User Trust

**Risk**: Users don't trust RL suggestions, reject them by default

**Mitigation**:
- Start with assistant mode (user decides)
- Show confidence scores
- Provide explanations
- Gradual rollout with A/B testing
- Allow easy opt-out to traditional mode

### Risk 4: Poor Generalization

**Risk**: Agent overfits to training data, fails on new UCs

**Mitigation**:
- Regularization techniques (dropout, L2)
- Cross-validation during training
- Test on held-out domains
- Ensemble methods (multiple agents voting)

### Risk 5: Reinforcing Biases

**Risk**: Agent learns human biases from feedback

**Mitigation**:
- Regular audits for bias
- Diverse training data
- Fairness constraints in training
- Expert review of agent decisions

---

## 10. Technology Stack & Tools

### 10.1 RL Frameworks

**Recommended**: Stable-Baselines3 (SB3)
- Well-maintained
- Easy to use
- Production-ready
- Good documentation

**Alternatives**:
- Ray RLlib (for distributed training)
- TF-Agents (TensorFlow-based)
- PyTorch RL libraries

### 10.2 Infrastructure

```yaml
Development:
  - Python 3.9+
  - PyTorch 2.0+
  - Stable-Baselines3
  - SQLite (data collection)
  - Jupyter (experimentation)

Production:
  - Model serving: TorchServe or ONNX Runtime
  - Storage: PostgreSQL (scale beyond SQLite)
  - Monitoring: Prometheus + Grafana
  - Logging: ELK stack
  - CI/CD: GitHub Actions

Cloud (optional):
  - Training: AWS SageMaker or Google Colab
  - Storage: S3 or Cloud Storage
  - Compute: GPU instances for training
```

### 10.3 Development Tools

```python
# requirements_rl.txt
torch>=2.0.0
stable-baselines3>=2.0.0
gymnasium>=0.29.0
tensorboard>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

# For model serving
onnx>=1.14.0
onnxruntime>=1.15.0

# For experimentation
jupyter>=1.0.0
ipywidgets>=8.0.0
```

---

## 11. Expected Outcomes

### Short Term (3-6 months)

1. **Data Collection Complete**
   - 1000+ analyzed UCs with user feedback
   - 10,000+ individual decision feedbacks
   - Baseline metrics established

2. **Initial RL Agents Trained**
   - Controller Selection Agent operational
   - Protection Function Agent operational
   - 70-80% accuracy on test set

3. **Assistant Mode Deployed**
   - Users can get RL suggestions
   - Feedback loop established
   - Initial user studies completed

### Medium Term (6-12 months)

1. **Auto Mode Operational**
   - RL agents make decisions automatically
   - 80-90% accuracy
   - 30-50% reduction in manual effort

2. **Multi-Domain Support**
   - Agents trained for 3+ domains
   - Transfer learning working
   - Domain-specific fine-tuning

3. **Continuous Learning Pipeline**
   - Online learning from user corrections
   - Periodic retraining automated
   - Model versioning and A/B testing

### Long Term (12+ months)

1. **Production-Grade System**
   - 90%+ accuracy on all decision types
   - Sub-second inference time
   - Highly trusted by users

2. **Advanced Features**
   - Personalized agents per user/team
   - Explainable AI (why this decision?)
   - Curriculum learning for rare patterns

3. **Business Impact**
   - 50%+ faster UC analysis
   - Higher quality diagrams
   - Increased user satisfaction
   - Competitive advantage

---

## 12. Next Steps

### Immediate Actions (This Week)

1. **Review and Validate Plan**
   - Get stakeholder buy-in
   - Identify potential issues
   - Refine timeline

2. **Setup Logging Infrastructure**
   - Add decision logging to analyzer
   - Create experience database schema
   - Implement basic feedback collection

3. **Create Synthetic Training Data**
   - Generate simple UC examples
   - Create ground truth analyses
   - Bootstrap initial dataset

### Next Month

1. **Implement Feedback UI**
   - Star rating system
   - Accept/Reject buttons
   - Correction interface

2. **Deploy Data Collection Version**
   - Release to beta users
   - Monitor data collection
   - Fix issues

3. **Begin Offline Training Prep**
   - Preprocess collected data
   - Implement first RL agent (Controller Selection)
   - Setup training pipeline

---

## 13. Conclusion

Reinforcement Learning can significantly enhance the UC-Analyzer by:

1. **Learning from Experience**: Continuously improve from user feedback
2. **Adapting to Users**: Learn user preferences and domain patterns
3. **Reducing Manual Effort**: Automate more decisions accurately
4. **Scaling Quality**: Maintain high quality as UC complexity grows

**Key Success Factors**:
- Sufficient training data (collect early and continuously)
- User trust (transparent, explainable, optional)
- Robust fallbacks (traditional mode always available)
- Continuous monitoring (detect and fix issues quickly)

**This is a long-term investment** that will pay off through:
- Faster UC analysis
- Higher quality diagrams
- Better user experience
- Competitive differentiation

The plan is ambitious but achievable with proper execution and commitment.

**Start small, learn fast, scale gradually.**
