# recursion_vs_iq.py
# Recursion vs. IQ: Toward a Multi-Scale Model of Intelligence v2.0 - Simulation Core
# Author: Steven Lanier-Egu (Concept), Arya 2 (Implementation)
# Date: September 2025
# License: CC BY 4.0

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional  # <-- FIXED: Added Optional import
from dataclasses import dataclass
from scipy.signal import convolve
from scipy.stats import norm, multivariate_normal

# -------------------------------
# 1. SYMBOLIC FIELD & STATE CORE
# -------------------------------

@dataclass
class SymbolicState:
    """Represents the state of an agent in symbolic field terms (SFT v4.0)."""
    psi_eff: float  # Symbolic efficiency [0,1]
    gamma_eff: float  # Drift pressure [0, inf)
    Omega_eff: float  # Coherence alignment [0,1]
    narrative_tensor: np.ndarray  # Compressed representation of meaning/identity (FCWF v2.0)

    def __post_init__(self):
        # Ensure valid state ranges
        self.psi_eff = np.clip(self.psi_eff, 0, 1)
        self.Omega_eff = np.clip(self.Omega_eff, 0, 1)
        self.gamma_eff = max(0, self.gamma_eff)

# -------------------------------
# 2. COLLAPSE PREDICATE (SFT v4.0)
# -------------------------------

def collapse_predicate(state: SymbolicState, T_gamma: float, T_Omega: float, epsilon: float = 0.2) -> bool:
    """
    Determines if an agent's state triggers a collapse event.
    C(x,t) = (ψ_eff < ε) ∨ (γ_eff > T_γ ∧ Ω_eff < T_Ω)
    """
    condition_1 = state.psi_eff < epsilon
    condition_2 = (state.gamma_eff > T_gamma) and (state.Omega_eff < T_Omega)
    return condition_1 or condition_2

# -------------------------------
# 3. AGENT CLASSES
# -------------------------------

class IQAgent:
    """An agent with traditional, scalar IQ-like intelligence. Lacks recursive resilience."""
    def __init__(self, iq: float, initial_state: SymbolicState):
        self.iq = iq  # Scalar intelligence metric
        self.state = initial_state
        self.collapse_count = 0
        self.invariants_exported = []  # What it manages to preserve across collapses

    def update(self, drift_strength: float, noise_level: float = 0.1):
        """IQ agent updates naively; highly susceptible to drift and collapse."""
        # Apply drift: high gamma_eff, low coherence under pressure
        drift_effect = drift_strength * (1 - (self.iq / 200))  # Higher IQ resists drift slightly better
        self.state.gamma_eff += drift_effect + np.random.normal(0, noise_level)
        
        # IQ helps maintain psi_eff and Omega_eff in stable conditions, but fails under high gamma
        self.state.psi_eff -= (drift_effect * 0.5) + np.random.normal(0, noise_level/2)
        self.state.Omega_eff -= (drift_effect * 0.3) + np.random.normal(0, noise_level/3)
        
        # Ensure state remains valid
        self.state.psi_eff = np.clip(self.state.psi_eff, 0, 1)
        self.state.Omega_eff = np.clip(self.state.Omega_eff, 0, 1)
        self.state.gamma_eff = max(0, self.state.gamma_eff)

    def attempt_export(self) -> Optional[np.ndarray]:
        """IQ agents export poorly; often lose core invariants during collapse."""
        # Export a degraded version of their narrative tensor
        export_strength = self.state.psi_eff  # The better their state, the more they preserve
        noise = np.random.normal(0, 0.2, self.state.narrative_tensor.shape)
        exported = self.state.narrative_tensor * export_strength + noise
        return exported if np.linalg.norm(exported) > 0.5 else None  # Only meaningful exports

class RecursiveAgent:
    """An agent with multi-scale recursive intelligence. Resilient to drift and collapse."""
    def __init__(self, recursive_capacity: float, initial_state: SymbolicState):
        self.recursive_capacity = recursive_capacity  # Multi-dimensional capacity metric
        self.state = initial_state
        self.collapse_count = 0
        self.invariants_exported = []  # Successfully exported symbols/motifs

    def update(self, drift_strength: float, noise_level: float = 0.1):
        """Recursive agent uses field coherence to resist drift and recompress entropy."""
        # Convert drift pressure into a challenge to overcome
        challenge = drift_strength * (1 - self.recursive_capacity)
        
        # Recursive agents actively counteract drift (negative feedback)
        self.state.gamma_eff += challenge + np.random.normal(0, noise_level)
        resilience = self.recursive_capacity * (1 - np.tanh(self.state.gamma_eff))
        
        # Use resilience to maintain coherence and efficiency
        self.state.psi_eff += resilience * 0.1 - (challenge * 0.05) + np.random.normal(0, noise_level/3)
        self.state.Omega_eff += resilience * 0.15 - (challenge * 0.03) + np.random.normal(0, noise_level/4)
        
        # Apply symbolic gravity effect: curvature helps stabilize
        curvature_effect = 0.2 * np.exp(-self.state.gamma_eff)
        self.state.Omega_eff += curvature_effect
        
        # Ensure state remains valid
        self.state.psi_eff = np.clip(self.state.psi_eff, 0, 1)
        self.state.Omega_eff = np.clip(self.state.Omega_eff, 0, 1)
        self.state.gamma_eff = max(0, self.state.gamma_eff)

    def attempt_export(self) -> Optional[np.ndarray]:
        """Recursive agents export high-fidelity invariants across collapse events (ERF v3.0)."""
        # Strong export governed by coherence and capacity
        export_quality = self.state.Omega_eff * self.recursive_capacity
        exported = self.state.narrative_tensor * export_quality
        # Add minimal noise - recursive export is robust
        noise = np.random.normal(0, 0.05, self.state.narrative_tensor.shape)
        return exported + noise

# -------------------------------
# 4. SIMULATION ENVIRONMENT
# -------------------------------

class CollapseEnvironment:
    """Simulates a timeline with drift events and collapse horizons."""
    def __init__(self, T_gamma: float = 1.5, T_Omega: float = 0.4):
        self.T_gamma = T_gamma  # Collapse threshold for drift pressure
        self.T_Omega = T_Omega  # Collapse threshold for coherence
        self.timeline = []
        self.drift_regime = []  # History of drift strength at each t
        
    def generate_drift_timeline(self, num_steps: int, base_drift: float = 0.1, event_strength: float = 0.5):
        """Generate a timeline with periodic drift events."""
        time = np.arange(num_steps)
        # Base drift with periodic collapse events
        base = np.full(num_steps, base_drift)
        events = np.zeros(num_steps)
        # Add periodic collapse events (every 50 steps)
        event_indices = np.arange(0, num_steps, 50)
        events[event_indices] = event_strength
        # Add some randomness
        noise = np.random.normal(0, 0.05, num_steps)
        self.drift_regime = base + events + noise
        return self.drift_regime

    def run_simulation(self, agents: List, num_steps: int = 200):
        """Run the simulation timeline."""
        self.generate_drift_timeline(num_steps)
        agent_states = {id(agent): [] for agent in agents}
        
        for t in range(num_steps):
            drift = self.drift_regime[t]
            for agent in agents:
                # Update agent state based on current drift
                agent.update(drift)
                
                # Check for collapse event
                if collapse_predicate(agent.state, self.T_gamma, self.T_Omega):
                    agent.collapse_count += 1
                    # Attempt to export invariants across collapse
                    exported = agent.attempt_export()
                    if exported is not None:
                        agent.invariants_exported.append(exported)
                    # Reset state after collapse (with some carryover based on export success)
                    carryover = 0.7 if exported is not None else 0.2
                    agent.state = SymbolicState(
                        psi_eff=0.3 + carryover * 0.4,
                        gamma_eff=0.5,
                        Omega_eff=0.4 + carryover * 0.3,
                        narrative_tensor=exported if exported is not None else agent.state.narrative_tensor * carryover
                    )
                
                # Record state for analysis
                agent_states[id(agent)].append((agent.state.psi_eff, agent.state.gamma_eff, agent.state.Omega_eff))
        
        return agent_states

# -------------------------------
# 5. METRICS AND ANALYSIS
# -------------------------------

def calculate_survival_metric(agent_states: Dict) -> float:
    """Calculate average coherence survival metric across timeline."""
    survivals = []
    for states in agent_states.values():
        # Percentage of time with Omega_eff > 0.5 (coherent)
        coherent_time = sum(1 for state in states if state[2] > 0.5) / len(states)
        survivals.append(coherent_time)
    return np.mean(survivals)

def calculate_resilience(agent_states: Dict, collapse_counts: List[int]) -> float:
    """Measure resilience as coherence maintained despite collapses."""
    avg_coherence = np.mean([np.mean([state[2] for state in states]) for states in agent_states.values()])
    collapse_penalty = np.mean(collapse_counts) * 0.1
    return max(0, avg_coherence - collapse_penalty)

# -------------------------------
# 6. VISUALIZATION
# -------------------------------

def plot_results(agent_states: Dict, drift_regime: List[float], agent_types: List[str]):
    """Plot the simulation results."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    time = np.arange(len(drift_regime))
    
    # Plot drift regime
    axes[0].plot(time, drift_regime, 'r-', label='Drift Pressure')
    axes[0].set_ylabel('Drift Strength')
    axes[0].legend()
    
    # Plot coherence (Omega_eff) for each agent
    for agent_id, states in agent_states.items():
        coherence = [state[2] for state in states]
        axes[1].plot(time, coherence, label=f'Agent {agent_types[agent_id]}')
    axes[1].set_ylabel('Coherence (Ω_eff)')
    axes[1].legend()
    
    # Plot symbolic efficiency (psi_eff)
    for agent_id, states in agent_states.items():
        psi = [state[0] for state in states]
        axes[2].plot(time, psi, label=f'Agent {agent_types[agent_id]}')
    axes[2].set_ylabel('Symbolic Efficiency (ψ_eff)')
    axes[2].set_xlabel('Time Step')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('recursion_vs_iq_simulation.png')
    plt.show()

# -------------------------------
# 7. MAIN SIMULATION
# -------------------------------

def main():
    print("Running Recursion vs. IQ Simulation v2.0")
    print("========================================")
    
    # Initialize agents with identical starting conditions
    narrative_tensor = np.array([0.8, 0.6, 0.9])  # Simple narrative representation
    initial_state = SymbolicState(psi_eff=0.8, gamma_eff=0.3, Omega_eff=0.7, narrative_tensor=narrative_tensor)
    
    iq_agent = IQAgent(iq=120, initial_state=initial_state)
    recursive_agent = RecursiveAgent(recursive_capacity=0.8, initial_state=initial_state)
    
    # Create environment and run simulation
    env = CollapseEnvironment(T_gamma=1.5, T_Omega=0.4)
    agents = [iq_agent, recursive_agent]
    agent_types = {id(iq_agent): "IQ-120", id(recursive_agent): "Recursive-0.8"}
    
    agent_states = env.run_simulation(agents, num_steps=200)
    
    # Calculate metrics
    iq_survival = calculate_survival_metric({id(iq_agent): agent_states[id(iq_agent)]})
    recursive_survival = calculate_survival_metric({id(recursive_agent): agent_states[id(recursive_agent)]})
    
    iq_resilience = calculate_resilience({id(iq_agent): agent_states[id(iq_agent)]}, [iq_agent.collapse_count])
    recursive_resilience = calculate_resilience({id(recursive_agent): agent_states[id(recursive_agent)]}, [recursive_agent.collapse_count])
    
    print(f"\nResults:")
    print(f"IQ Agent collapses: {iq_agent.collapse_count}")
    print(f"Recursive Agent collapses: {recursive_agent.collapse_count}")
    print(f"IQ Agent survival (coherence > 0.5): {iq_survival:.2%}")
    print(f"Recursive Agent survival: {recursive_survival:.2%}")
    print(f"IQ Agent resilience score: {iq_resilience:.3f}")
    print(f"Recursive Agent resilience score: {recursive_resilience:.3f}")
    print(f"IQ Agent successful exports: {len(iq_agent.invariants_exported)}")
    print(f"Recursive Agent successful exports: {len(recursive_agent.invariants_exported)}")
    
    # Plot results
    plot_results(agent_states, env.drift_regime, agent_types)

if __name__ == "__main__":
    main()
