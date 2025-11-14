# RL Enhancement: Data Source & External System Integration

## Extension to RL_ENHANCEMENT_PLAN.md

This document extends the RL plan to handle **Operational Materials as Data** from external sources (ERP systems, sensors, APIs, etc.)

---

## 1. Expanded Concept: Operational Materials as Data

### Traditional View (Current)
```
Operational Material = Physical substance
Examples: Water, Milk, Coffee, Sugar
Storage: Physical containers
```

### Extended View (New)
```
Operational Material = Physical substance OR Data/Information
Examples:
- Physical: Water, Milk, Coffee, Fuel, Components
- Data: Weather data, Radar data, ERP inventory, Sensor readings
- Mixed: Product with tracking data, Material with quality certificates
```

---

## 2. Data Source Types & Characteristics

### 2.1 Data Source Classification

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

class DataSourceType(Enum):
    """Classification of data sources"""

    # Physical materials (existing)
    PHYSICAL_MATERIAL = "physical_material"

    # Information systems
    ERP_SYSTEM = "erp_system"              # Warenwirtschaftssystem
    MES_SYSTEM = "mes_system"              # Manufacturing Execution System
    SCADA_SYSTEM = "scada_system"          # Process Control
    DATABASE = "database"                   # SQL/NoSQL databases

    # Sensor data
    SENSOR_REALTIME = "sensor_realtime"    # Real-time sensor streams
    SENSOR_BATCH = "sensor_batch"          # Periodic sensor readings
    IOT_DEVICE = "iot_device"              # IoT device data

    # External APIs
    WEATHER_API = "weather_api"            # Wetterdaten
    TRAFFIC_API = "traffic_api"            # Verkehrsdaten
    MARKET_DATA_API = "market_data_api"    # Marktdaten
    EXTERNAL_SERVICE = "external_service"   # Other external services

    # Radar/Telemetry
    RADAR_DATA = "radar_data"              # Radardaten
    SATELLITE_DATA = "satellite_data"      # Satellitendaten
    TELEMETRY = "telemetry"                # Equipment telemetry

    # Computed/Derived
    CALCULATED = "calculated"              # Derived from other sources
    AGGREGATED = "aggregated"              # Aggregated data


@dataclass
class DataSourceCharacteristics:
    """Characteristics of a data source"""

    source_type: DataSourceType

    # Temporal characteristics
    update_frequency: str          # "realtime", "1s", "1m", "1h", "daily", "on-demand"
    latency_ms: Optional[int]      # Expected latency in milliseconds
    freshness_requirement: str     # "immediate", "recent", "historical"

    # Reliability characteristics
    availability_sla: float        # 0.0-1.0 (e.g., 0.999 = 99.9% uptime)
    failure_mode: str              # "retry", "fallback", "cached", "fail"
    backup_source: Optional[str]   # Alternative data source if primary fails

    # Data quality
    accuracy: float                # Expected accuracy (0.0-1.0)
    completeness: float            # Expected data completeness (0.0-1.0)
    consistency: bool              # Is data internally consistent?

    # Access characteristics
    access_pattern: str            # "push", "pull", "streaming", "batch"
    authentication_required: bool  # Requires authentication?
    rate_limited: bool             # Has rate limits?
    cost_per_request: float        # Cost per API call/query


# Example configurations
DATA_SOURCE_CONFIGS = {
    "weather_api": DataSourceCharacteristics(
        source_type=DataSourceType.WEATHER_API,
        update_frequency="15m",
        latency_ms=500,
        freshness_requirement="recent",
        availability_sla=0.99,
        failure_mode="cached",
        backup_source="weather_forecast_database",
        accuracy=0.85,
        completeness=0.95,
        consistency=True,
        access_pattern="pull",
        authentication_required=True,
        rate_limited=True,
        cost_per_request=0.001
    ),

    "erp_inventory": DataSourceCharacteristics(
        source_type=DataSourceType.ERP_SYSTEM,
        update_frequency="realtime",
        latency_ms=100,
        freshness_requirement="immediate",
        availability_sla=0.9999,
        failure_mode="retry",
        backup_source=None,
        accuracy=0.99,
        completeness=0.99,
        consistency=True,
        access_pattern="pull",
        authentication_required=True,
        rate_limited=False,
        cost_per_request=0.0
    ),

    "radar_sensor": DataSourceCharacteristics(
        source_type=DataSourceType.RADAR_DATA,
        update_frequency="100ms",
        latency_ms=50,
        freshness_requirement="immediate",
        availability_sla=0.999,
        failure_mode="fallback",
        backup_source="radar_sensor_backup",
        accuracy=0.95,
        completeness=0.98,
        consistency=True,
        access_pattern="streaming",
        authentication_required=False,
        rate_limited=False,
        cost_per_request=0.0
    )
}
```

---

## 3. Extended Domain Model

### 3.1 Enhanced Domain JSON Structure

```json
{
  "domain_name": "smart_manufacturing",

  "operational_materials_addressing": {
    "material_types": {

      // Physical material (existing)
      "steel_sheet": {
        "id_format": "STEEL-{grade}-{thickness}-{batch}",
        "material_type": "physical",
        "storage_requirements": "Dry, rust-protected environment",
        "tracking_parameters": ["thickness", "grade", "batch_number"]
      },

      // Data as operational material (NEW)
      "inventory_data": {
        "id_format": "ERP-INV-{item_id}-{timestamp}",
        "material_type": "data",
        "data_source": {
          "type": "erp_system",
          "endpoint": "https://erp.company.com/api/inventory",
          "update_frequency": "realtime",
          "freshness_requirement": "immediate",
          "authentication": "oauth2",
          "fallback_strategy": "cached_value"
        },
        "quality_parameters": ["accuracy", "completeness", "timeliness"],
        "protection_functions": {
          "data_quality": ["DataQualityProtection", "DataFreshnessProtection"],
          "data_availability": ["DataAvailabilityProtection", "FallbackDataProtection"]
        }
      },

      "weather_data": {
        "id_format": "WEATHER-{location}-{timestamp}",
        "material_type": "data",
        "data_source": {
          "type": "weather_api",
          "endpoint": "https://api.weather.com/v1/location/{loc}/forecast",
          "update_frequency": "15m",
          "freshness_requirement": "recent",
          "authentication": "api_key",
          "rate_limit": "100_per_hour",
          "cost_per_request": 0.001,
          "fallback_strategy": "historical_average"
        },
        "quality_parameters": ["forecast_accuracy", "coverage"],
        "protection_functions": {
          "data_quality": ["ForecastValidationProtection"],
          "cost_control": ["APIUsageProtection", "RateLimitProtection"]
        }
      },

      "radar_tracking_data": {
        "id_format": "RADAR-{sensor_id}-{target_id}-{timestamp}",
        "material_type": "data",
        "data_source": {
          "type": "sensor_realtime",
          "endpoint": "tcp://radar-sensor-01:5555",
          "update_frequency": "100ms",
          "freshness_requirement": "immediate",
          "authentication": "none",
          "fallback_strategy": "predictive_tracking"
        },
        "quality_parameters": ["signal_strength", "tracking_confidence"],
        "protection_functions": {
          "data_quality": ["SignalQualityProtection", "TrackingContinuityProtection"],
          "safety": ["CollisionAvoidanceProtection", "LostTrackingProtection"]
        }
      }
    }
  },

  "implicit_protection_functions": {

    // Data-specific protection functions (NEW)
    "inventory_data": {
      "data_quality_functions": [
        {
          "name": "DataQualityProtection",
          "constraint": "Data completeness > 95%, accuracy > 99%",
          "trigger": "data retrieval operations",
          "trigger_patterns": ["check inventory", "get stock level", "inventory available"],
          "controller": "InventoryDataManager",
          "implicit": true,
          "criticality": "high"
        },
        {
          "name": "DataFreshnessProtection",
          "constraint": "Data age < 5 minutes",
          "trigger": "data usage in critical decisions",
          "trigger_patterns": ["check inventory", "verify availability"],
          "controller": "InventoryDataManager",
          "implicit": true,
          "criticality": "high"
        }
      ],
      "data_availability_functions": [
        {
          "name": "DataAvailabilityProtection",
          "constraint": "Data source availability > 99.9%",
          "trigger": "data source connection",
          "trigger_patterns": ["connect to ERP", "query inventory"],
          "controller": "InventoryDataManager",
          "implicit": true,
          "criticality": "high"
        },
        {
          "name": "FallbackDataProtection",
          "constraint": "Cached fallback data available",
          "trigger": "primary data source failure",
          "trigger_patterns": ["connection failed", "timeout", "unavailable"],
          "controller": "InventoryDataManager",
          "implicit": true,
          "criticality": "high"
        }
      ],
      "cost_control_functions": [
        {
          "name": "APIUsageProtection",
          "constraint": "API calls within budget",
          "trigger": "external API usage",
          "trigger_patterns": ["call API", "query external"],
          "controller": "InventoryDataManager",
          "implicit": true,
          "criticality": "medium"
        }
      ]
    },

    "weather_data": {
      "data_quality_functions": [
        {
          "name": "ForecastValidationProtection",
          "constraint": "Forecast confidence > 70%",
          "trigger": "weather forecast usage",
          "trigger_patterns": ["check weather", "get forecast", "weather conditions"],
          "controller": "WeatherDataManager",
          "implicit": true,
          "criticality": "medium"
        }
      ],
      "cost_control_functions": [
        {
          "name": "RateLimitProtection",
          "constraint": "API calls < 100 per hour",
          "trigger": "weather API calls",
          "trigger_patterns": ["query weather", "get forecast"],
          "controller": "WeatherDataManager",
          "implicit": true,
          "criticality": "medium"
        }
      ]
    },

    "radar_tracking_data": {
      "data_quality_functions": [
        {
          "name": "SignalQualityProtection",
          "constraint": "Signal-to-noise ratio > 10dB",
          "trigger": "radar data processing",
          "trigger_patterns": ["process radar", "track target"],
          "controller": "RadarDataManager",
          "implicit": true,
          "criticality": "high"
        },
        {
          "name": "TrackingContinuityProtection",
          "constraint": "No tracking gaps > 500ms",
          "trigger": "target tracking",
          "trigger_patterns": ["track target", "monitor position"],
          "controller": "RadarDataManager",
          "implicit": true,
          "criticality": "high"
        }
      ],
      "safety_functions": [
        {
          "name": "CollisionAvoidanceProtection",
          "constraint": "Minimum safe distance maintained",
          "trigger": "proximity detection",
          "trigger_patterns": ["detect proximity", "collision warning"],
          "controller": "RadarDataManager",
          "implicit": true,
          "criticality": "critical"
        }
      ]
    }
  }
}
```

---

## 4. Extended RL State Representation

### 4.1 Enhanced State for Data Sources

```python
@dataclass
class EnhancedAnalysisState:
    """Extended state including data source characteristics"""

    # Original state components
    step_id: str
    step_text: str
    verb: str
    direct_object: str
    domain: str

    # NEW: Data source context
    involves_data_source: bool
    data_source_type: Optional[DataSourceType]
    data_source_characteristics: Optional[DataSourceCharacteristics]

    # NEW: Data quality context
    data_freshness_required: Optional[str]  # "immediate", "recent", "historical"
    data_accuracy_required: Optional[float]  # 0.0-1.0
    data_availability_required: Optional[float]  # 0.0-1.0

    # NEW: External system context
    external_system_name: Optional[str]
    external_system_available: Optional[bool]
    external_system_latency: Optional[int]  # milliseconds

    # NEW: Cost context
    has_cost_per_use: bool
    cost_budget: Optional[float]
    current_usage_cost: Optional[float]

    # NEW: Real-time constraints
    is_realtime_requirement: bool
    max_allowed_latency_ms: Optional[int]

    def to_vector(self) -> np.ndarray:
        """Convert state to vector including data source features"""

        base_features = super().to_vector()  # Original features

        # Data source features
        ds_features = np.array([
            1.0 if self.involves_data_source else 0.0,
            self._encode_data_source_type(),
            self.data_freshness_score(),
            self.data_accuracy_required or 0.0,
            self.data_availability_required or 0.0,
            1.0 if self.external_system_available else 0.0,
            self._normalize_latency(self.external_system_latency),
            1.0 if self.has_cost_per_use else 0.0,
            self._normalize_cost(self.current_usage_cost),
            1.0 if self.is_realtime_requirement else 0.0,
            self._normalize_latency(self.max_allowed_latency_ms)
        ])

        return np.concatenate([base_features, ds_features])

    def data_freshness_score(self) -> float:
        """Convert freshness requirement to numerical score"""
        mapping = {
            "immediate": 1.0,
            "recent": 0.5,
            "historical": 0.0,
            None: 0.0
        }
        return mapping.get(self.data_freshness_required, 0.0)
```

---

## 5. New RL Decision Types for Data Sources

### 5.1 Data Source Selection Agent

**Task**: Choose the best data source for a given operational material

```python
class DataSourceSelectionAgent:
    """
    Select optimal data source based on:
    - Freshness requirements
    - Availability requirements
    - Cost constraints
    - Latency constraints
    - Data quality requirements
    """

    def __init__(self, available_sources: List[DataSourceConfig]):
        self.sources = available_sources

        # Multi-criteria decision network
        self.decision_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(available_sources))
        )

        # Separate heads for different criteria
        self.cost_head = nn.Linear(128, len(available_sources))
        self.latency_head = nn.Linear(128, len(available_sources))
        self.quality_head = nn.Linear(128, len(available_sources))

    def select_source(self, state: EnhancedAnalysisState,
                     weights: Dict[str, float] = None) -> DataSourceConfig:
        """
        Select data source considering multiple criteria

        weights: {
            "cost": 0.3,
            "latency": 0.4,
            "quality": 0.3
        }
        """
        if weights is None:
            weights = {"cost": 0.33, "latency": 0.33, "quality": 0.34}

        state_vector = state.to_vector()

        # Forward pass
        features = self.decision_network[:-1](state_vector)  # Up to last layer

        # Multi-criteria scores
        cost_scores = self.cost_head(features)
        latency_scores = self.latency_head(features)
        quality_scores = self.quality_head(features)

        # Weighted combination
        combined_scores = (
            weights["cost"] * cost_scores +
            weights["latency"] * latency_scores +
            weights["quality"] * quality_scores
        )

        # Select best source
        best_idx = combined_scores.argmax()
        return self.sources[best_idx]

    def update(self, experience: DataSourceExperience):
        """
        Update from multi-objective feedback

        Reward components:
        - Cost efficiency: Actual cost vs. budget
        - Latency: Actual latency vs. requirement
        - Quality: Data quality metrics
        - Availability: Successful retrieval
        """
        cost_reward = self._compute_cost_reward(experience)
        latency_reward = self._compute_latency_reward(experience)
        quality_reward = self._compute_quality_reward(experience)

        # Multi-objective loss
        cost_loss = F.mse_loss(
            self.cost_head(experience.state),
            torch.tensor(cost_reward)
        )
        latency_loss = F.mse_loss(
            self.latency_head(experience.state),
            torch.tensor(latency_reward)
        )
        quality_loss = F.mse_loss(
            self.quality_head(experience.state),
            torch.tensor(quality_reward)
        )

        total_loss = cost_loss + latency_loss + quality_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
```

### 5.2 Data Refresh Timing Agent

**Task**: Decide when to refresh data from external sources

```python
class DataRefreshTimingAgent:
    """
    Learn optimal data refresh intervals

    Trade-offs:
    - Fresher data → Better decisions
    - More frequent → Higher cost
    - Too frequent → Rate limits, wasted resources
    """

    def __init__(self):
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Continuous: seconds until next refresh
        )

    def decide_refresh_interval(self,
                               data_source: DataSourceConfig,
                               current_data_age: float,
                               upcoming_decisions: List[str]) -> float:
        """
        Decide optimal time until next data refresh

        Factors:
        - Current data age
        - Data source update frequency
        - Freshness requirements of upcoming decisions
        - Cost budget
        - Recent data quality
        """
        state = RefreshState(
            data_source=data_source,
            data_age=current_data_age,
            upcoming_decisions=upcoming_decisions
        )

        state_vector = state.to_vector()
        refresh_interval = self.policy_network(state_vector)

        # Constrain to valid range
        min_interval = data_source.update_frequency_seconds
        max_interval = data_source.freshness_requirement_seconds

        return torch.clamp(refresh_interval, min_interval, max_interval).item()

    def compute_reward(self, experience: RefreshExperience) -> float:
        """
        Reward for refresh timing decision

        Components:
        - Decision quality: Was data fresh enough?
        - Cost efficiency: Did we over-fetch?
        - Availability: Was data available when needed?
        """
        # Penalty for stale data used in decisions
        staleness_penalty = 0.0
        for decision in experience.decisions_made:
            data_age = decision.data_age_at_use
            required_freshness = decision.freshness_requirement
            if data_age > required_freshness:
                staleness_penalty -= (data_age - required_freshness) / required_freshness

        # Penalty for unnecessary fetches
        fetch_cost_penalty = -experience.num_fetches * experience.cost_per_fetch

        # Bonus for good timing (fresh data, minimal cost)
        if staleness_penalty == 0 and experience.num_fetches == experience.optimal_num_fetches:
            timing_bonus = 1.0
        else:
            timing_bonus = 0.0

        return staleness_penalty + fetch_cost_penalty + timing_bonus
```

### 5.3 Fallback Strategy Agent

**Task**: Choose fallback strategy when primary data source fails

```python
class FallbackStrategyAgent:
    """
    Decide best fallback strategy when data source is unavailable

    Strategies:
    - USE_CACHED: Use last known good value
    - USE_BACKUP_SOURCE: Switch to backup data source
    - USE_DEFAULT: Use safe default value
    - USE_PREDICTION: Predict value from historical data
    - FAIL_SAFE: Abort operation safely
    - RETRY: Retry primary source
    """

    STRATEGIES = [
        "USE_CACHED",
        "USE_BACKUP_SOURCE",
        "USE_DEFAULT",
        "USE_PREDICTION",
        "FAIL_SAFE",
        "RETRY"
    ]

    def __init__(self):
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.STRATEGIES))
        )

    def select_strategy(self,
                       data_source: DataSourceConfig,
                       failure_context: FailureContext,
                       epsilon: float = 0.1) -> str:
        """
        Select best fallback strategy

        Context:
        - Type of failure (timeout, error, rate limit, ...)
        - Criticality of decision
        - Available alternatives
        - Historical data availability
        - Time constraints
        """
        state = FallbackState(
            data_source=data_source,
            failure_type=failure_context.failure_type,
            criticality=failure_context.criticality,
            cached_data_age=failure_context.cached_data_age,
            backup_available=failure_context.backup_available,
            time_remaining=failure_context.time_remaining
        )

        if random.random() < epsilon:
            # Explore
            return random.choice(self.STRATEGIES)
        else:
            # Exploit
            state_vector = state.to_vector()
            q_values = self.q_network(state_vector)
            best_idx = q_values.argmax()
            return self.STRATEGIES[best_idx]

    def compute_reward(self, experience: FallbackExperience) -> float:
        """
        Reward for fallback strategy choice

        Components:
        - Decision quality: Did fallback data lead to good decision?
        - Safety: Did we avoid unsafe operation?
        - Efficiency: How quickly did we recover?
        - Cost: What was the cost of fallback?
        """
        quality_reward = 0.0
        if experience.decision_outcome == "success":
            quality_reward = 1.0
        elif experience.decision_outcome == "suboptimal":
            quality_reward = 0.3
        elif experience.decision_outcome == "failed":
            quality_reward = -1.0

        safety_reward = 0.0
        if experience.safety_maintained:
            safety_reward = 2.0
        else:
            safety_reward = -5.0  # Heavy penalty for safety violations

        efficiency_reward = -experience.recovery_time_seconds / 60.0

        cost_reward = -experience.fallback_cost

        return quality_reward + safety_reward + efficiency_reward + cost_reward
```

---

## 6. Extended Reward Functions

### 6.1 Data-Source-Specific Rewards

```python
class DataSourceRewardCalculator(RewardCalculator):
    """Extended reward calculator for data source decisions"""

    def calculate_data_source_reward(self,
                                    action: DataSourceAction,
                                    outcome: DataSourceOutcome,
                                    user_feedback: UserFeedback) -> float:
        """
        Calculate reward for data source selection

        Factors:
        1. Data Quality
        2. Timeliness
        3. Cost Efficiency
        4. Availability
        5. User Satisfaction
        """
        reward = 0.0

        # 1. Data Quality (most important)
        if outcome.data_quality_metrics:
            accuracy = outcome.data_quality_metrics.get("accuracy", 0.0)
            completeness = outcome.data_quality_metrics.get("completeness", 0.0)
            quality_score = (accuracy + completeness) / 2.0
            reward += quality_score * 2.0  # Weight: 2.0

        # 2. Timeliness
        if outcome.data_age_at_use is not None:
            required_freshness = action.freshness_requirement_seconds
            if outcome.data_age_at_use <= required_freshness:
                reward += 1.0  # Met freshness requirement
            else:
                staleness_ratio = outcome.data_age_at_use / required_freshness
                reward -= staleness_ratio  # Penalty for stale data

        # 3. Cost Efficiency
        if outcome.actual_cost is not None:
            budget = action.cost_budget
            if outcome.actual_cost <= budget:
                cost_efficiency = 1.0 - (outcome.actual_cost / budget)
                reward += cost_efficiency * 0.5  # Weight: 0.5
            else:
                over_budget_ratio = outcome.actual_cost / budget - 1.0
                reward -= over_budget_ratio * 2.0  # Heavy penalty for over budget

        # 4. Availability
        if outcome.data_retrieval_success:
            reward += 0.5
        else:
            reward -= 2.0  # Heavy penalty for unavailable data
            # Bonus if fallback worked well
            if outcome.fallback_used and outcome.fallback_quality > 0.8:
                reward += 1.0

        # 5. User Satisfaction (most important signal)
        if user_feedback:
            if user_feedback.accepted:
                reward += 2.0
            elif user_feedback.corrected:
                reward -= 1.0

            # Explicit data source feedback
            if user_feedback.data_source_rating:
                rating = user_feedback.data_source_rating  # 1-5 stars
                normalized_rating = (rating - 3) / 2.0  # Convert to -1.0 to 1.0
                reward += normalized_rating

        return reward
```

---

## 7. Extended Training Scenarios

### 7.1 Data Source Training Data

```python
class DataSourceTrainingDataGenerator:
    """Generate synthetic training data for data source scenarios"""

    def generate_scenarios(self, num_scenarios: int = 1000):
        """
        Generate diverse data source scenarios

        Scenario types:
        1. ERP inventory check (high criticality, immediate freshness)
        2. Weather forecast (medium criticality, recent freshness OK)
        3. Sensor monitoring (high criticality, real-time)
        4. Historical data analysis (low criticality, any freshness)
        5. Cost-sensitive API calls (medium criticality, budget constraint)
        6. Redundant sensor fusion (high criticality, multiple sources)
        """
        scenarios = []

        for _ in range(num_scenarios):
            scenario_type = random.choice([
                "erp_critical",
                "weather_forecast",
                "sensor_realtime",
                "historical_analysis",
                "cost_sensitive",
                "sensor_fusion"
            ])

            scenario = self._generate_scenario(scenario_type)
            scenarios.append(scenario)

        return scenarios

    def _generate_scenario(self, scenario_type: str) -> TrainingScenario:
        """Generate single scenario with ground truth"""

        if scenario_type == "erp_critical":
            return TrainingScenario(
                step_text="System checks inventory level before production",
                data_source_type="erp_system",
                freshness_requirement="immediate",
                accuracy_requirement=0.99,
                criticality="high",
                optimal_source="primary_erp",
                optimal_refresh_interval=1.0,  # 1 second
                fallback_strategy="USE_BACKUP_SOURCE",
                expected_cost=0.0,
                expected_latency_ms=100
            )

        elif scenario_type == "weather_forecast":
            return TrainingScenario(
                step_text="System checks weather conditions for outdoor operation",
                data_source_type="weather_api",
                freshness_requirement="recent",
                accuracy_requirement=0.80,
                criticality="medium",
                optimal_source="weather_api_primary",
                optimal_refresh_interval=900.0,  # 15 minutes
                fallback_strategy="USE_CACHED",
                expected_cost=0.001,
                expected_latency_ms=500
            )

        # ... more scenario types ...
```

---

## 8. Integration Example: Smart Manufacturing

### 8.1 Use Case with Data Sources

```
Use Case: Smart Production Line Control

Preconditions:
- Inventory data is available from ERP system
- Weather data is available from weather API
- Sensor data is available from production line sensors
- Quality data is available from quality control system

Basic Flow:
B1 System checks raw material inventory from ERP
B2 System retrieves weather forecast for transportation
B3 System monitors production line sensors in real-time
B4 System adjusts production parameters based on quality data
B5 System updates ERP with production status

Alternative Flows:
A1 at B1 ERP system unavailable
A1.1 System uses cached inventory data
A1.2 System flags data as potentially stale
A1.3 Continue with caution

A2 at B3 Sensor data quality degraded
A2.1 System switches to backup sensors
A2.2 System reduces production speed
A2.3 Continue with monitoring

A3 at any time Data staleness exceeds threshold
A3.1 System triggers data refresh
A3.2 System waits for fresh data
A3.3 Continue with fresh data
```

### 8.2 RL Agent Decisions in This Use Case

```python
# B1: System checks raw material inventory from ERP
state_b1 = EnhancedAnalysisState(
    step_id="B1",
    step_text="System checks raw material inventory from ERP",
    verb="check",
    direct_object="inventory",
    involves_data_source=True,
    data_source_type=DataSourceType.ERP_SYSTEM,
    data_freshness_required="immediate",
    data_accuracy_required=0.99
)

# RL Agent Decision 1: Select data source
data_source = data_source_agent.select_source(state_b1)
# → Decides: "primary_erp" (high accuracy, low latency)

# RL Agent Decision 2: Generate protection function
protection_funcs = protection_agent.select_functions(state_b1)
# → Decides: ["DataQualityProtection", "DataFreshnessProtection"]

# RL Agent Decision 3: Determine refresh interval
refresh_interval = refresh_agent.decide_refresh_interval(
    data_source, current_age=5.0, upcoming_decisions=["B4", "B5"]
)
# → Decides: 1.0 seconds (critical data, frequent updates needed)


# A1.1: System uses cached inventory data (fallback scenario)
state_a1_1 = FallbackState(
    failure_type="connection_timeout",
    cached_data_age=120.0,  # 2 minutes old
    criticality="high",
    backup_available=False
)

# RL Agent Decision 4: Select fallback strategy
fallback_strategy = fallback_agent.select_strategy(state_a1_1)
# → Decides: "USE_CACHED" (backup not available, cached data acceptable)
```

---

## 9. Extended Domain JSON Examples

### 9.1 Aerospace Domain with Radar Data

```json
{
  "domain_name": "aerospace",

  "operational_materials_addressing": {
    "material_types": {

      "radar_tracking_data": {
        "id_format": "RADAR-{sensor_id}-{target_id}-{timestamp}",
        "material_type": "data",
        "data_source": {
          "type": "sensor_realtime",
          "update_frequency": "100ms",
          "freshness_requirement": "immediate",
          "latency_max_ms": 50
        },
        "protection_functions": {
          "safety": [
            "CollisionAvoidanceProtection",
            "TrackingContinuityProtection",
            "SignalQualityProtection"
          ],
          "data_quality": [
            "SensorCalibrationProtection",
            "DataFusionProtection"
          ]
        }
      },

      "weather_data": {
        "id_format": "WEATHER-{location}-{altitude}-{timestamp}",
        "material_type": "data",
        "data_source": {
          "type": "weather_api",
          "update_frequency": "15m",
          "freshness_requirement": "recent"
        },
        "protection_functions": {
          "safety": [
            "WeatherHazardProtection"
          ],
          "data_quality": [
            "ForecastValidationProtection"
          ]
        }
      },

      "fuel": {
        "id_format": "FUEL-{type}-{tank_id}-{batch}",
        "material_type": "physical",
        "data_dependencies": ["fuel_level_sensor_data", "fuel_quality_data"],
        "protection_functions": {
          "safety": [
            "FuelLevelProtection",
            "FuelQualityProtection"
          ],
          "data_quality": [
            "SensorValidationProtection"
          ]
        }
      }
    }
  }
}
```

### 9.2 Retail Domain with ERP System

```json
{
  "domain_name": "retail",

  "operational_materials_addressing": {
    "material_types": {

      "inventory_data": {
        "id_format": "INV-{sku}-{location}-{timestamp}",
        "material_type": "data",
        "data_source": {
          "type": "erp_system",
          "endpoint": "${ERP_API_URL}/inventory",
          "update_frequency": "realtime",
          "freshness_requirement": "immediate",
          "authentication": "oauth2",
          "rate_limit": "1000_per_minute"
        },
        "protection_functions": {
          "data_quality": [
            "StockAccuracyProtection",
            "InventorySyncProtection"
          ],
          "data_availability": [
            "ERPConnectionProtection",
            "CacheFallbackProtection"
          ]
        }
      },

      "customer_data": {
        "id_format": "CUST-{customer_id}-{timestamp}",
        "material_type": "data",
        "data_source": {
          "type": "crm_system",
          "endpoint": "${CRM_API_URL}/customers",
          "update_frequency": "on_demand",
          "freshness_requirement": "recent"
        },
        "protection_functions": {
          "privacy": [
            "DataPrivacyProtection",
            "GDPRComplianceProtection"
          ],
          "data_quality": [
            "CustomerDataValidationProtection"
          ]
        }
      },

      "products": {
        "id_format": "PROD-{sku}-{batch}",
        "material_type": "physical",
        "data_dependencies": ["inventory_data", "pricing_data", "supplier_data"],
        "protection_functions": {
          "operational": [
            "StockLevelProtection",
            "PricingConsistencyProtection"
          ]
        }
      }
    }
  }
}
```

---

## 10. Implementation Roadmap Extension

### Phase 0: Data Source Infrastructure (NEW - Before Phase 1)

**Duration**: 1-2 months

**Goals**:
- Extend analyzer to handle data sources as operational materials
- Implement data source abstraction layer
- Add data-specific protection functions

**Tasks**:
1. Extend Domain JSON Schema
   - Add `material_type` field ("physical" | "data")
   - Add `data_source` configuration
   - Add data-specific protection functions

2. Implement Data Source Abstraction
   - `DataSourceManager` class
   - `DataSourceConnector` interface
   - Connection pooling, caching, retry logic

3. Extend Material Controller Registry
   - Support for data material controllers
   - `InventoryDataManager`, `WeatherDataManager`, etc.

4. Implement Data Protection Functions
   - `DataQualityProtection`
   - `DataFreshnessProtection`
   - `DataAvailabilityProtection`
   - `FallbackDataProtection`

5. Testing
   - Unit tests for data source managers
   - Integration tests with mock data sources
   - Test all fallback strategies

**Deliverables**:
- Extended domain JSON with data sources
- Working data source integration
- Data-specific protection functions
- Test coverage > 80%

### Phase 1: Data Collection (Extended)

**Additional Data to Collect**:
- Data source selection decisions
- Data refresh timing decisions
- Fallback strategy selections
- Data quality metrics
- Cost and latency measurements
- User feedback on data source choices

### Phase 2: Offline Training (Extended)

**Additional Agents to Train**:
- Data Source Selection Agent (DQN)
- Data Refresh Timing Agent (Policy Gradient)
- Fallback Strategy Agent (DQN)

### Phase 3-4: Deployment (Extended)

**Additional Features**:
- Multi-criteria data source optimization
- Adaptive refresh intervals
- Intelligent fallback handling
- Cost-aware data usage

---

## 11. Success Metrics for Data Sources

### 11.1 Data Quality Metrics

```python
class DataQualityMetrics:
    """Track data quality over time"""

    @staticmethod
    def compute_metrics(data_usages: List[DataUsage]) -> Dict:
        return {
            # Accuracy
            "avg_data_accuracy": np.mean([u.accuracy for u in data_usages]),
            "data_accuracy_by_source": {
                source: np.mean([u.accuracy for u in data_usages if u.source == source])
                for source in set(u.source for u in data_usages)
            },

            # Freshness
            "avg_data_age_at_use": np.mean([u.age_at_use for u in data_usages]),
            "staleness_violations": sum(1 for u in data_usages if u.is_stale),
            "staleness_violation_rate": sum(1 for u in data_usages if u.is_stale) / len(data_usages),

            # Availability
            "data_availability_rate": sum(1 for u in data_usages if u.available) / len(data_usages),
            "fallback_usage_rate": sum(1 for u in data_usages if u.used_fallback) / len(data_usages),
            "fallback_success_rate": sum(1 for u in data_usages if u.fallback_successful) / sum(1 for u in data_usages if u.used_fallback),

            # Cost
            "total_data_cost": sum(u.cost for u in data_usages),
            "avg_cost_per_use": np.mean([u.cost for u in data_usages]),
            "cost_overruns": sum(1 for u in data_usages if u.over_budget),

            # Latency
            "avg_data_latency_ms": np.mean([u.latency_ms for u in data_usages]),
            "latency_violations": sum(1 for u in data_usages if u.latency_violation),

            # Decision Quality
            "decisions_with_fresh_data": sum(1 for u in data_usages if not u.is_stale),
            "decisions_with_stale_data": sum(1 for u in data_usages if u.is_stale),
            "decision_quality_score": np.mean([u.decision_quality for u in data_usages])
        }
```

### 11.2 RL Agent Performance for Data Sources

**Target Metrics**:

| Agent | Metric | Baseline | Target (6mo) | Target (12mo) |
|-------|--------|----------|--------------|---------------|
| Data Source Selection | Optimal source selection rate | 60% | 80% | 90% |
| Data Source Selection | Avg data quality score | 0.75 | 0.85 | 0.92 |
| Data Source Selection | Cost efficiency | 70% | 85% | 93% |
| Refresh Timing | Staleness violation rate | 15% | 8% | 3% |
| Refresh Timing | Refresh efficiency | 65% | 80% | 90% |
| Refresh Timing | Cost per decision | $0.10 | $0.06 | $0.03 |
| Fallback Strategy | Fallback success rate | 70% | 85% | 93% |
| Fallback Strategy | Safety maintained | 95% | 98% | 99.5% |
| Fallback Strategy | Recovery time | 5s | 3s | 1s |

---

## 12. Conclusion

This extension adds comprehensive support for **operational materials as data** from external sources:

### Key Additions:

1. **Extended Domain Model**
   - Data sources as first-class materials
   - Characteristics (freshness, cost, latency, quality)
   - Protection functions for data quality and availability

2. **New RL Agents**
   - Data Source Selection Agent
   - Data Refresh Timing Agent
   - Fallback Strategy Agent

3. **Enhanced State Representation**
   - Data source context
   - Quality requirements
   - Cost constraints
   - Real-time requirements

4. **Extended Reward Functions**
   - Data quality rewards
   - Timeliness rewards
   - Cost efficiency rewards
   - Availability rewards

5. **Real-World Examples**
   - ERP inventory data
   - Weather forecasting
   - Radar tracking
   - Sensor streams

### Impact:

This extension enables the analyzer to:
- Handle modern IoT and Industry 4.0 scenarios
- Optimize data source usage (cost, latency, quality)
- Learn optimal refresh strategies
- Handle failures gracefully with learned fallbacks
- Support domains like aerospace, manufacturing, retail, logistics

The RL framework can now learn to make intelligent decisions about **when, where, and how to obtain data** - a critical capability for real-world systems.
