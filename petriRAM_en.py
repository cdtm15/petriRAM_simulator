#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: PhD(C) Cristian Tobar and PHD Mariela Muñoz
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heapq
import seaborn as sns
import math, numpy as np, heapq
from functools import partial
try:
    from scipy.stats import lognorm as _lognorm_dist
    from scipy.stats import gamma as _gamma_dist
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# Definición de lugares y transiciones
places = ['Idle', 'Running', 'Replacing', 'Failure', 'Repairing', 'Waiting']

transitions = ['Start', 'Complete', 'Preventive maintenance',
               'Preventive Maintenance Time', 'Failure_tran',
               'Corrective maintenance', 'Corrective Maintenance Time']

#Pre and Post matrices
pre_matrix = [
    [1, 0, 1, 0, 0, 0, 0],  # Idle
    [0, 1, 0, 0, 1, 0, 0],  # Running
    [0, 0, 0, 1, 0, 0, 0],  # Replacing
    [0, 0, 0, 0, 0, 1, 0],  # Fail
    [0, 0, 0, 0, 0, 0, 1],  # Repairing
    [0, 0, 0, 0, 0, 0, 1]   # Waiting/Hold
]

post_matrix = [
    [0, 1, 0, 1, 0, 0, 1],  # Idle
    [1, 0, 0, 0, 0, 0, 0],  # Running
    [0, 0, 1, 0, 0, 0, 0],  # Replacing
    [0, 0, 0, 0, 1, 0, 0],  # Fail
    [0, 0, 0, 0, 0, 1, 0],  # Repairing
    [0, 0, 0, 0, 1, 0, 0]   # Waiting/Hold
]

pre = pd.DataFrame(pre_matrix, index=places, columns=transitions)
post = pd.DataFrame(post_matrix, index=places, columns=transitions)

transitions_info = {
    "Start": {"type": "signal"},
    "Complete": {"type": "stochastic"},
    "Failure_tran": {"type": "stochastic"},
    "Corrective maintenance": {"type": "signal"},
    "Corrective Maintenance Time": {"type": "stochastic"},
    "Preventive maintenance": {"type": "signal"},
    "Preventive Maintenance Time": {"type": "stochastic"}
}

def is_enabled(transition, marking):
    """
   Check whether a given transition is enabled in the current marking.

   A transition is enabled if, for every place, the number of tokens 
   available is greater than or equal to the number of tokens required 
   by the pre-incidence matrix.

   Args:
       transition (str): The transition to check.
       marking (dict): The current marking, mapping each place to its token count.

   Returns:
       bool: True if the transition is enabled, False otherwise.
   """
    return all(marking.get(pl, 0) >= pre.loc[pl, transition] for pl in places)

def fire(transition, marking):
    """
    Fire a given transition, updating the marking accordingly.

    When a transition fires, tokens are removed from its input places
    (according to the pre-incidence matrix) and added to its output
    places (according to the post-incidence matrix).

    Args:
        transition (str): The transition to fire.
        marking (dict): The current marking, mapping each place to its token count.

    Returns:
        dict: The updated marking after firing the transition.
    """
    
    for pl in places:
        marking[pl] -= pre.loc[pl, transition]
        marking[pl] += post.loc[pl, transition]
    return marking

def _std_norm_cdf(z: float) -> float:
    
    """
    Standard normal CDF Φ(z), computed via the error function (erf).

    It returns the probability that a
    standard normal N(0,1) random variable is ≤ z.

    Args:
        z (float): Evaluation point.

    Returns:
        float: Φ(z) in [0, 1].
    """
    
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def _gamma_cdf_integer_k(t: float, k: int, theta: float) -> float:
    
    """
    Gamma/Erlang CDF for integer shape k (k >= 1), without SciPy.

    Implements the Erlang closed form:
        F(t; k, θ) = 1 - e^{-x} * Σ_{n=0}^{k-1} x^n / n!, where x = t/θ

    Args:
        t (float): Time (t >= 0). Returns 0.0 if t <= 0.
        k (int): Integer shape parameter (k >= 1).
        theta (float): Scale parameter (θ > 0).

    Returns:
        float: CDF value F(t) in [0, 1].
    """
    
    if t <= 0: return 0.0
    x = t / theta
    s = 0.0; e = math.exp(-x); term = 1.0
    for n in range(k):
        if n > 0: term *= x / n
        s += term
    return 1.0 - e * s

def failure_cdf(t: float, cfg: dict) -> float:
    """
   Unified failure-time CDF F(t) for Weibull, lognormal, or gamma models.

   Selects the distribution via parameter["dist"] => {"weibull", "lognormal", "gamma"}
   and evaluates F(t) using the parameters in the dictionary. The function returns 0.0
   for t <= 0. For unknown distributions, 0.0 is returned. If SciPy is available,
   it is used for lognormal and gamma; otherwise, an Erlang fallback is used for
   gamma with integer shape, and a closed-form normal CDF is used for lognormal.

   Args:
       t (float): Time at which to evaluate the CDF (t >= 0).
       cfg (dict): Distribution configuration. Expected keys by distribution:
           - Weibull:
               {"dist": "weibull", "alpha": <shape>, "scale": <scale>}
               F(t) = 1 - exp(- (t/scale)^alpha)
           - Lognormal (parameters in log-space):
               {"dist": "lognormal", "mu": <mu>, "sigma": <sigma>}
               F(t) = Φ((ln t - mu)/sigma), with Φ the standard normal CDF
               (SciPy path: lognorm.cdf with s=sigma, scale=exp(mu))
           - Gamma:
               {"dist": "gamma", "k": <shape> or "shape": <k>, "theta": <scale> or "scale": <theta>}
               (SciPy path: gamma.cdf with a=k, scale=theta)
               (No-SciPy fallback: Erlang CDF for integer k)

   Returns:
       float: CDF value F(t) in [0, 1]. Returns NaN if gamma is requested
              without SciPy and k is non-integer.

   Notes:
       - This function does not mutate `cfg`.
       - For lognormal without SciPy, a closed-form standard normal CDF is used.
       - For gamma without SciPy, only integer k (Erlang) is supported.
   """
    
    if t <= 0: return 0.0
    dist = cfg.get("dist", "weibull").lower()
    if dist == "weibull":
        alpha = cfg["alpha"]; scale = cfg["scale"]
        return 1.0 - math.exp(- (t / scale) ** alpha)
    elif dist == "lognormal":
        mu = cfg["mu"]; sigma = cfg["sigma"]
        if _HAVE_SCIPY:
            return float(_lognorm_dist.cdf(t, s=sigma, scale=math.exp(mu)))
        z = (math.log(t) - mu) / sigma
        return _std_norm_cdf(z)
    elif dist == "gamma":
        k = cfg.get("k") or cfg.get("shape")
        theta = cfg.get("theta") or cfg.get("scale")
        if _HAVE_SCIPY:
            return float(_gamma_dist.cdf(t, a=k, scale=theta))
        if int(k) == k and k > 0:
            return _gamma_cdf_integer_k(t, int(k), theta)
        return float("nan")
    return 0.0


def evaluate_failures_cdf_weibull_with_clock(clock, components, markings, operating_time, last_exec_time, event_queue, params_per_component):
    """
    Evaluate the probability of failure for each component based on the Weibull CDF,
    using the actual elapsed time since the last time it was in the 'Running' state.

    For every component currently in the 'Running' state:
        - Compute the elapsed time since the last evaluation.
        - Accumulate this time into its operating life.
        - Calculate the Weibull cumulative distribution function (CDF).
        - With probability F(t), schedule a failure event and reset its operating time.

    Args:
        clock (float): Current simulation time.
        components (list): List of component identifiers.
        markings (dict): Current marking for each component (places and token counts).
        operating_time (dict): Accumulated operating time for each component.
        last_exec_time (dict): Last evaluation time for each component.
        event_queue (list): Priority queue of scheduled events.
        params_per_component (dict): Weibull distribution parameters per component.

    Returns:
        None: Updates are applied directly to `markings`, `operating_time`, 
        `last_exec_time`, and `event_queue`.
    """
    for c in components:
       if markings[c]["Running"] > 0:
           delta_time = clock - last_exec_time.get(c, clock)
           last_exec_time[c] = clock
           operating_time[c] += delta_time

           cfg = params_per_component[c]["Failure_tran"]
           F_t = failure_cdf(operating_time[c], cfg)
           u = np.random.rand()
           if (not np.isnan(F_t)) and (u < F_t):
               heapq.heappush(event_queue, (clock, "Failure_tran", c))
               operating_time[c] = 0

def schedule_event(current_time, transition, comp, params_per_component):
    """
    Compute the scheduled (absolute) time for a stochastic transition.

    This function looks up the distribution parameters for the given
    transition and component, and returns the absolute timestamp at
    which the event should occur based on the current simulation time.

    Behavior (per transition type/label):
    - For "Complete": draws from a normal distribution with parameters
      like {"loc": ..., "scale": ...} and returns current_time + N(loc, scale).
    - For "Corrective Maintenance Time" and "Preventive Maintenance Time":
      draws from a uniform distribution with parameters "low" and"high"
      and returns current_time + U(low, high).

    Args:
        current_time (float): Current simulation time.
        transition (str): Transition label to schedule (e.g., "Complete").
        comp (str): Component identifier.
        params_per_component (dict): Parameters per component and transition

    Returns:
        float: The absolute time at which the event should occur.
    """
    p = params_per_component[comp].get(transition, {})
    if transition == "Complete":
        return current_time + np.random.normal(**p)
    if transition in ["Corrective Maintenance Time", "Preventive Maintenance Time"]:
        return current_time + np.random.uniform(**p)
    return current_time

def execute_periodic_start(clock, markings, components, fired_transitions, states_over_time, last_exec_time):
    """
    Fire the 'Start' transition for all components currently in the 'Idle' place.
    Updates the marking, logs fired transitions, and records states at the current time.

    For each component:
      - If it has a token in 'Idle', move one token from 'Idle' to 'Running'.
      - Update the component's last execution time to the current clock value.
      - Append the fired transition ('Start') to `fired_transitions`.
      - Record the component's non-zero places at this timestamp in `states_over_time`.

    Args:
        clock (float): Current simulation time.
        markings (dict): Markings per component.
        components (list): List of component identifiers.
        fired_transitions (list): Log list to append tuples (clock, component, transition).
        states_over_time (list): Log list to append tuples (clock, component, place) for non-zero places.
        last_exec_time (dict): Dict to store the last evaluation/exec time per component.

    Returns:
        None: The updates are applied in-place to `markings`, `fired_transitions`,
        `states_over_time`, and `last_exec_time`.
    """
    for c in components:
        if markings[c]["Idle"] > 0:
            markings[c]["Idle"] -= 1
            markings[c]["Running"] += 1
            last_exec_time[c] = clock  # keep track of the last time it entered 'Running'
            fired_transitions.append((clock, c, "Start"))
            for place in places:
                if markings[c][place] > 0:
                    states_over_time.append((clock, c, place))

def fire_failure_tran_updated(markings, comp, components, clock):
    """
    Custom handler for the 'Failure_tran' transition.

    - Moves only the failing component's token from 'Running' to 'Failure'.
    - Sends every *other* component to 'Waiting':
        * If it was 'Running', move one token from 'Running' to 'Waiting'.
        * Else if it was 'Idle', move one token from 'Idle' to 'Waiting'.
    - Returns both the updated markings and a list of (time, component, place)
      tuples capturing the state changes, useful for logging/plotting.

    Args:
        markings (dict): Per-component markings
        comp (str): Identifier of the component that fails.
        components (list): List of all component identifiers.
        clock (float): Current simulation time.

    Returns:
        tuple:
            - dict: The updated markings.
            - list[tuple]: A list of state updates as (clock, component, place).
    """
    updated_states = []

    # The failing component goes to 'Failure'
    m = markings[comp]
    m["Running"] -= 1
    m["Failure"] += 1
    updated_states.append((clock, comp, "Failure"))

    # All others are sent to 'Waiting'
    for other in components:
        if other != comp:
            mo = markings[other]
            if mo["Running"] > 0:
                mo["Running"] -= 1
                mo["Waiting"] += 1
                updated_states.append((clock, other, "Waiting"))
            elif mo["Idle"] > 0:
                mo["Idle"] -= 1
                mo["Waiting"] += 1
                updated_states.append((clock, other, "Waiting"))

    return markings, updated_states

def fire_corrective_maint_time(markings, failed_component, components):
    """
    Handle the completion of the 'Corrective Maintenance Time' transition.

    Behavior:
      - The failed component finishes repair: move one token from 'Repairing' to 'Idle'.
      - For every other component currently 'Waiting', move one token from 'Waiting' to 'Idle'.

    Notes:
      - Updates are applied in-place to `markings`.

    Args:
        markings (dict): Per-component markings.
        failed_component (str): Identifier of the component that was under repair.
        components (list[str]): List of all component identifiers.
    """
    m_fail = markings[failed_component]
    m_fail["Repairing"] -= 1
    m_fail["Idle"] += 1

    for other in components:
        if other != failed_component:
            mo = markings[other]
            if mo["Waiting"] > 0:
                mo["Waiting"] -= 1
                mo["Idle"] += 1
    
    return markings

def fire_preventive_maint_time(markings, failed_component, components):
    """
    Handle the completion of the 'Preventive Maintenance Time' transition.

    Behavior:
      - The target component finishes replacement: move one token from 'Replacing' to 'Idle'.
      - For every other component currently 'Waiting', move one token from 'Waiting' to 'Idle'.
      - Returns the updated `markings` (mirrors the original function's return behavior).
                                        
      Notes:
        - Updates are applied in-place to `markings`.

    Args:
        markings (dict): Per-component markings.
        failed_component (str): Identifier of the component that was being replaced.
        components (list[str]): List of all component identifiers.
    """
    m_fail = markings[failed_component]
    m_fail["Replacing"] -= 1
    m_fail["Idle"] += 1

    for other in components:
        if other != failed_component:
            mo = markings[other]
            if mo["Waiting"] > 0:
                mo["Waiting"] -= 1
                mo["Idle"] += 1

    return markings

def fire_preventive_maintenance(markings, comp, components, clock):
    """
    Fire the 'Preventive Maintenance' transition for the selected component.

    Behavior:
      - The target component leaves 'Idle' and enters 'Replacing'.
      - All other components:
        * If in 'Running', move one token from 'Running' to 'Waiting'.
        * If in 'Idle', move one token from 'Idle' to 'Waiting'.
      - Records all state changes with their timestamp.

    Args:
        markings (dict): Current marking of all components (tokens per place).
        comp (str): Component undergoing preventive maintenance.
        components (list[str]): List of all component identifiers.
        clock (float): Current simulation time.

    Returns:
        tuple:
            - dict: Updated markings.
            - list[tuple]: A list of (clock, component, new_state) representing
              the changes in marking.
    """
    updated_states = []

    # The selected component enters 'Replacing'
    m = markings[comp]
    m["Idle"] -= 1
    m["Replacing"] += 1
    updated_states.append((clock, comp, "Replacing"))

    # All other components enter 'Waiting'
    for other in components:
        if other != comp:
            mo = markings[other]
            if mo["Running"] > 0:
                mo["Running"] -= 1
                mo["Waiting"] += 1
                updated_states.append((clock, other, "Waiting"))
            elif mo["Idle"] > 0:
                mo["Idle"] -= 1
                mo["Waiting"] += 1
                updated_states.append((clock, other, "Waiting"))

    return markings, updated_states
           
def try_schedule_all_stochastic(clock, comp, markings, queue, params_per_component):
    """
    Schedule all stochastic transitions for a given component.

    For each transition labeled as "stochastic" in `transitions_info`, this function:
      1) Computes the absolute event time via `schedule_event`.
      2) Pushes the tuple (event_time, transition, component) into the priority queue.

    Notes:
      - This function does not mutate `markings`.
      - It relies on the global dictionaries/lists: `transitions` and `transitions_info`.
      - The queue is assumed to be a heap-based priority queue (e.g., `heapq`).

    Args:
        clock (float): Current simulation time.
        comp (str): Component identifier.
        markings (dict): Current markings (not modified here).
        queue (list): Priority queue (heap) where events will be pushed.
        params_per_component (dict): Parameters per component/transition, used by `schedule_event`.

    Returns:
        None
    """
    for t in transitions:
        if transitions_info[t]["type"] == "stochastic":
            event_time = schedule_event(clock, t, comp, params_per_component)
            heapq.heappush(queue, (event_time, t, comp))

def simulate_multicolor_petri(T_MAX, num_components, params_per_component, num_pm_cycles):
    """
   Main multicolor Petri net simulation loop.

   Initializes per-component markings and logs, then advances the simulation clock,
   scheduling and firing transitions as dictated by the model (including periodic 'Start',
   completion, failures, and maintenance events). Returns two DataFrames:
   a transition log and a state-over-time log.

   Args:
       T_MAX (float): Maximum simulation time (inclusive).
       num_components (int): Number of components to simulate (labels C1, C2, ...).
       params_per_component (dict): Transition parameters per component (e.g., durations).
       num_pm_cycles (int): Number of preventive maintenance cycles.

   Returns:
       tuple[pd.DataFrame, pd.DataFrame]:
           - df_transitions: columns ["Time", "Component", "Transition"].
           - df_states: columns ["Time", "Component", "State"].
   """
    
    components = [f"C{i+1}" for i in range(num_components)]
    technician_arrival_time = 1000
    start_interval = 5000
    clock = 0
    operating_time = {c: 0 for c in components}
    last_exec_time = {c: 0 for c in components}
    event_queue = []
 
    # Initial state
    markings = {c: {pl: 0 for pl in places} for c in components}
    for c in components:
        markings[c]["Idle"] = 1
 
    states_over_time = []
    fired_transitions = []
    completed_cycles = {c: 0 for c in components}
    
    # Programar "Start" cada cierto time
    time = 0
    while time <= T_MAX:
        heapq.heappush(event_queue, (time, "Start", "ALL"))
        time += start_interval
        
    # for c in components:
    #     try_schedule_all_stochastic(clock, c, markings, event_queue)
    for c in components:
        try_schedule_all_stochastic(clock, c, markings, event_queue, params_per_component)

    while event_queue and clock <= T_MAX:
        clock, t, comp = heapq.heappop(event_queue)
        
        # Evaluar fallos antes de procesar el evento actual
        evaluate_failures_cdf_weibull_with_clock(
            clock, components, markings, operating_time, last_exec_time, event_queue, params_per_component
        )

        if t == "Start" and comp == "ALL":
            execute_periodic_start(clock, markings, components, fired_transitions, states_over_time, last_exec_time)
            continue
        

        m = markings[comp]

        if t == "Failure_tran" and is_enabled(t, m):
            markings, nuevos_estados = fire_failure_tran_updated(markings, comp, components, clock)
            fired_transitions.append((clock, comp, t))
            states_over_time.extend(nuevos_estados)
            heapq.heappush(event_queue, (clock + technician_arrival_time, "Corrective maintenance", comp))
            
        elif t == "Corrective maintenance" and is_enabled(t, m):
            fire(t, m)
            fired_transitions.append((clock, comp, t))
            states_over_time.append((clock, comp, "Repairing"))
            heapq.heappush(event_queue, (schedule_event(clock, "Corrective Maintenance Time",comp, params_per_component), "Corrective Maintenance Time", comp))

            
        elif t == "Corrective Maintenance Time" and markings[comp]["Repairing"] > 0:
            markings = fire_corrective_maint_time(markings, comp, components)
            fired_transitions.append((clock, comp, t))

            for c in components:
                for place in places:
                    if markings[c][place] > 0:
                        states_over_time.append((clock, c, place))
            try_schedule_all_stochastic(clock, comp, markings, event_queue, params_per_component)
    
        elif t == "Complete" and is_enabled(t, m):
            fire(t, m)
            fired_transitions.append((clock, comp, t))
            states_over_time.append((clock, comp, "Idle"))
        
            completed_cycles[comp] += 1
            if completed_cycles[comp] >= num_pm_cycles:
                heapq.heappush(event_queue, (clock, "Preventive maintenance", comp))
                completed_cycles[comp] = 0
            try_schedule_all_stochastic(clock, comp, markings, event_queue, params_per_component)

        elif t == "Preventive maintenance" and is_enabled(t, m):
            markings, nuevos_estados = fire_preventive_maintenance(markings, comp, components, clock)
            fired_transitions.append((clock, comp, t))
            states_over_time.extend(nuevos_estados)
            heapq.heappush(event_queue, (schedule_event(clock, "Preventive Maintenance Time", comp, params_per_component), "Preventive Maintenance Time", comp))
            
        elif t == "Preventive Maintenance Time" and markings[comp]["Replacing"] > 0:
            markings = fire_preventive_maint_time(markings, comp, components)
            fired_transitions.append((clock, comp, t))
            for c in components:
                for place in places:
                    if markings[c][place] > 0:
                        states_over_time.append((clock, c, place))
            try_schedule_all_stochastic(clock, comp, markings, event_queue, params_per_component)

        # Evaluar posibles fallos por envejecimiento (CDF Weibull)
        #evaluar_fallos_cdf_weibull(clock, components, markings, operating_time, event_queue, params_per_component)
        #evaluate_failures_cdf_weibull_with_clock(clock, components, markings, operating_time, last_exec_time, event_queue, params_per_component)
    
    df_transitions = pd.DataFrame(fired_transitions, columns=["Time", "Component", "Transition"])
    df_states = pd.DataFrame(states_over_time, columns=["Time", "Component", "State"])
    return df_transitions, df_states

def monte_carlo_indicators(simulate_func, n_iterations):
    """
    Run the simulation multiple times and aggregate maintenance indicators.

    For each iteration:
      1) Calls `simulate_func()` which must return (_, df_states).
      2) Computes per-component indicators via `calculate_maintenance_indicators(df_states)`.
      3) Tags results with the iteration index and appends to a list.

    After all iterations:
      - Concatenates all per-iteration results into `df_results`.
      - Builds a per-component summary (`summary`) averaging metrics across iterations.
      - Builds an overall-average summary across components (`summary_avg`).
      - Produces a per-component wide view by iteration (`by_component_dict`)
        using `to_iteration_format(df_results)`.

    Expected columns in the per-iteration indicators DataFrame:
      ["Component", "Parameter/Indicator", "Mean", "Std", "SEM"]
      plus we add ["Iteration"] in this function.

    Args:
        simulate_func (callable): Function that runs ONE simulation and returns
            a tuple (_, df_states). The first element is ignored here.
        n_iterations (int): Number of Monte Carlo repetitions.

    Returns:
        tuple:
            - summary (pd.DataFrame): Aggregated per-component indicators across iterations,
              columns: ["Component", "Parameter/Indicator", "Mean", "Std", "SEM"].
            - df_results (pd.DataFrame): Long-format indicators with an "Iteration" column
              for every component/indicator/iteration.
            - summary_avg (pd.DataFrame): Overall average across components (if >1 component),
              same metric columns as `summary`, with "Component" set to "Average".
              If only one component exists, returns an empty DataFrame.
            - by_component_dict (dict[str, pd.DataFrame]): For each component, a wide table
              pivoted by "Iteration" (index) and "Parameter/Indicator" (columns) with "Mean" values.
              Includes an extra key "Average" if multiple components exist (row-wise average).
    """
    
    all_results = []

    for i in range(n_iterations):
        _, df_states = simulate_func()
        df_metrics = calculate_maintenance_indicators(df_states)
        df_metrics["Iteration"] = i + 1
        all_results.append(df_metrics)

    df_results = pd.concat(all_results, ignore_index=True)


    summary = (
        df_results.groupby(["Component", "Parameter/Indicator"])
        .agg(
            Mean=("Mean", "mean"),
            Minimum=("Mean", "min"),
            Maximum=("Mean", "max"),
            Std=("Mean", "std"),
            SEM=("Mean", lambda x: x.std(ddof=1) / np.sqrt(len(x)))
        )
        .reset_index()
    )

    # Calcular promedio general si hay más de un componente
    if summary["Component"].nunique() > 1:
        summary_avg = (
            summary
            .groupby("Parameter/Indicator")
            .agg({
                "Mean": "mean",
                "Minimum": "mean",
                "Maximum": "mean",
                "Std": "mean",
                "SEM": "mean"
            })
            .reset_index()
        )
        summary_avg.insert(0, "Component", "Average")
    else:
        summary_avg = pd.DataFrame()
    
    # Generar estructura adicional por componente en formato iteración
    by_component_dict = to_iteration_format(df_results)

    return summary, df_results, summary_avg, by_component_dict

def calculate_maintenance_indicators(df_states):
    """
    Compute maintenance indicators per component from a state-over-time log.

    Expected input:
        df_states: pd.DataFrame with columns ["Component", "Time", "State"]
                   States used by the model: "Idle", "Running", "Failure",
                   "Repairing", "Replacing", "Waiting".

    For each component, the function:
      - Sorts by time and builds (state, duration) segments as the time deltas
        between consecutive rows.
      - Defines availability states vs. unavailability states and computes:
          * MUT (up time) and MDT (down time) totals.
          * Availability (%) = MUT / (MUT + MDT)
      - Tracks transitions to compute:
          * Number of failures
          * MTTF: mean time of cumulative 'Running' duration immediately
                  preceding each 'Failure'
          * MLDT: mean duration from 'Failure' -> 'Repairing' (logistics)
          * MTTR: mean duration from 'Repairing' -> 'Idle' (repair)
          * MTBF: (MUT - sum(repair_times)) / number_of_failures  (same as original)
      - Identifies down events (contiguous segments of unavailability) to compute:
          * MDT (as the distribution of down event durations)
          * MTBDE: mean time between down events
          
    - Mission Reliability (R): missions completed without failure / missions attempted.
        We treat each contiguous 'Running' segment as a mission attempt and classify it as:
          * success if the next state after 'Running' is 'Idle'
          * failure if the next state after 'Running' is 'Failure'
        Segments that end in other states are ignored (neither success nor attempt)
        to avoid misclassifying logistics or external waits as mission outcomes.
      - Conditional Reliability (RC): same fraction but conditioned on being up at mission start.
        In this model, missions start in 'Running'.
      - Mean Maintenance Time (MMT): mean of maintenance durations for
        *either* corrective repair or preventive maintenance, excluding logistics delays,
        as in Raptor【turn3file9†L10-L17】.
        Here we use durations of 'Repairing' and 'Replacing' only.

    Returns:
        resultados_df: pd.DataFrame in long format with columns:
            ["Component", "Parameter/Indicator", "Mean", "Minimum",
             "Maximum", "Std", "SEM"]
    """
    resultados = []
    df_states = df_states.sort_values(by=["Component", "Time"])
    
    for component, df_comp in df_states.groupby("Component"):
        df_comp = df_comp.sort_values("Time")
        times = df_comp["Time"].values
        states = df_comp["State"].values
    
        durations = []      # list of tuples (state, duration)
        transitions = []    # list of tuples (prev_state, curr_state)
    
        # Build durations and transitions between consecutive timestamps
        for i in range(1, len(times)):
            dur = times[i] - times[i - 1]
            prev_state = states[i - 1]
            durations.append((prev_state, dur))
            transitions.append((prev_state, states[i]))
    
        df_dur = pd.DataFrame(durations, columns=["State", "Duration"])
    
        # Availability definition
        available_states = ["Idle", "Running"]
        unavailable_states = ["Failure", "Repairing", "Replacing", "Waiting"]
    
        mut = df_dur[df_dur["State"].isin(available_states)]["Duration"].sum()
        mdt = df_dur[df_dur["State"].isin(unavailable_states)]["Duration"].sum()
    
        availability = mut / (mut + mdt) if (mut + mdt) > 0 else 0
    
        failures = []
        repair_times = []
        pm_times = []
        logistics_times = []
        time_to_failure = []
    
        # Collect repair/PM/logistics durations and TTFs
        for i in range(1, len(transitions)):
            prev, curr = transitions[i]
            dur = durations[i][1]
    
            # Count failure and compute TTF as cumulative 'Running' immediately before a 'Failure'
            if curr == "Failure":
                failures.append(i)
                ttf = 0
                for j in range(i, -1, -1):
                    if durations[j][0] == "Running":
                        ttf += durations[j][1]
                    else:
                        break
                time_to_failure.append(ttf)
    
            # Logistics time: Failure -> Repairing
            if prev == "Failure" and curr == "Repairing":
                logistics_times.append(dur)
    
            # Repair time: Repairing -> Idle
            if prev == "Repairing" and curr == "Idle":
                repair_times.append(dur)
    
        # Preventive maintenance time: time spent in 'Replacing'
        pm_times = df_dur[df_dur["State"] == "Replacing"]["Duration"].tolist()
    
        n_failures = len(failures)
        mtbf = (mut - sum(repair_times)) / n_failures if n_failures > 0 else np.nan
        mttf = np.mean(time_to_failure) if time_to_failure else np.nan
        mttr = np.mean(repair_times) if repair_times else np.nan
        mldt = np.mean(logistics_times) if logistics_times else np.nan
    
        # Mission-based Reliability (R) & Conditional Reliability (RC)
        mission_attempts = 0
        mission_successes = 0
        for k in range(len(durations) - 1):
            st_k = durations[k][0]
            st_next = durations[k + 1][0]
            if st_k == "Running" and st_next in ("Idle", "Failure"):
                mission_attempts += 1
                if st_next == "Idle":
                    mission_successes += 1
        R_mission = (mission_successes / mission_attempts) if mission_attempts > 0 else np.nan
        RC_mission = R_mission  # starts 'up' by construction
    
        # Mean Maintenance Time (MMT): corrective repair or PM (no logistics)
        mmt_values = repair_times + pm_times  # keep the list to expose distribution
        # (mean will be computed by add_param)
    
        # Helper to append metrics
        def add_param(name, values):
            if values and len(values) > 0:
                resultados.append({
                    "Component": component,
                    "Parameter/Indicator": name,
                    "Mean": np.mean(values),
                    "Minimum": np.min(values),
                    "Maximum": np.max(values),
                    "Std": np.std(values, ddof=1) if len(values) > 1 else 0,
                    "SEM": (np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0
                })
            else:
                resultados.append({
                    "Component": component,
                    "Parameter/Indicator": name,
                    "Mean": np.nan,
                    "Minimum": np.nan,
                    "Maximum": np.nan,
                    "Std": np.nan,
                    "SEM": np.nan
                })
    
        # Scalars
        add_param("Availability (%)", [availability])
        add_param("MTBF", [mtbf])
        add_param("Mission Reliability (R)", [R_mission])  # per Raptor【turn2file2†L172-L182】
        add_param("Conditional Reliability (RC)", [RC_mission])  # per Raptor【turn2file2†L192-L201】
    
        # Distributions
        add_param("MTTF", time_to_failure)
        add_param("MTTR", repair_times)
        add_param("MLDT", logistics_times)
        add_param("MMT (Repair or PM)", mmt_values)  # excludes logistics【turn3file9†L10-L17】
    
        # Down events and MTBDE/MDT
        down_events = []
        times_between_downs = []
        in_down = False
        down_start = None
        last_down_index = None
    
        for i in range(len(durations)):
            state, dur = durations[i]
            if state in unavailable_states:
                if not in_down:
                    in_down = True
                    down_start = i
                    if last_down_index is not None:
                        time_between = sum(d[1] for d in durations[last_down_index:i])
                        times_between_downs.append(time_between)
                    last_down_index = i
            else:
                if in_down:
                    end = i
                    event_duration = sum(d[1] for d in durations[down_start:end])
                    down_events.append(event_duration)
                    in_down = False
    
        add_param("MTBDE", times_between_downs)
        add_param("MDT", down_events)
        add_param("Number of failures", [n_failures])
    
    resultados_df = pd.DataFrame(resultados)
    return resultados_df

def to_iteration_format(df_results):
    """
    Transform a long-format indicators DataFrame (by component and iteration)
    into a dict of wide tables, one per component, with indicators as columns.

    Expects columns:
        - "Component"
        - "Iteration"
        - "Parameter/Indicator"
        - "Mean"

    Returns:
        dict[str, pd.DataFrame]:
            { component_name: pivoted_df, ..., "Average": avg_df (if >1 components) }
    """
    components = df_results["Component"].unique()
    by_component = {}

    # One pivot (Iteration x Indicators) per component
    for comp in components:
        df_comp = df_results[df_results["Component"] == comp]
        df_pivot = (
            df_comp
            .pivot(index="Iteration", columns="Parameter/Indicator", values="Mean")
            .reset_index()
        )
        by_component[comp] = df_pivot

    # If there is more than one component, compute a row-wise average table
    if len(components) > 1:
        all_pivots = list(by_component.values())
        df_avg = all_pivots[0].copy()
        for df in all_pivots[1:]:
            # Merge on Iteration, keep left columns when overlapping; drop *_drop
            df_avg = df_avg.merge(df, on="Iteration", suffixes=(None, "_drop"))
            df_avg = df_avg.loc[:, ~df_avg.columns.str.endswith("_drop")]

        indicator_cols = [col for col in df_avg.columns if col != "Iteration"]
        # Element-wise mean across the aligned matrices
        df_avg[indicator_cols] = np.mean([df[indicator_cols] for df in all_pivots], axis=0)
        by_component["Average"] = df_avg

    return by_component

def plot_boxplots_per_indicator(df_long):
    """
    Generate a figure with boxplot subplots for each maintenance indicator,
    showing the distribution for each component.

    Notes:
        - Requires seaborn as `sns` and matplotlib.pyplot as `plt` to be imported.
        - The layout uses 2 columns and as many rows as needed (ceil division).

    Args:
        df_long (pd.DataFrame): Long-format DataFrame with columns
            ['Component', 'Parameter/Indicator', 'Mean'].

    Returns:
        None: Saves a PDF file ('components_comparison_boxplot.pdf') and shows the figure.
    """
    indicators = df_long["Parameter/Indicator"].unique()
    num_indicators = len(indicators)
    cols = 2
    rows = (num_indicators + cols - 1) // cols  # ceiling division
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten()

    for idx, indicator in enumerate(indicators):
        ax = axes[idx]
        subset = df_long[df_long["Parameter/Indicator"] == indicator]
        sns.boxplot(data=subset, x="Component", y="Mean", ax=ax)
        ax.set_title(indicator)
        ax.set_ylabel("Mean per Iteration")
        ax.set_xlabel("Component")

    # Hide any extra axes if there are more subplots than needed
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("components_comparison_boxplot.pdf")
    plt.show()

def plot_ram_summary(df_long, bins=20, savepath="summary_2x2_hist_box.pdf"):
    """
    Create a 2x2 figure:
      (1) Histogram of Mission Reliability (R) over iterations/components
      (2) Histogram of Availability (%)
      (3) Histogram of MMT (Repair or PM)
      (4) Boxplot of Number of failures by Component (distribution across iterations)

    Args:
        df_long (pd.DataFrame): Long-format indicators with columns including
            ["Component", "Parameter/Indicator", "Mean", "Iteration", ...].
        bins (int): Number of bins for histograms.
        savepath (str): Output PDF filename.

    Returns:
        None (shows the figure and saves to `savepath`)
    """
    # Helper to fetch a series by indicator name (robust to missing)
    def _get_series(ind_name):
        mask = df_long["Parameter/Indicator"] == ind_name
        if not mask.any():
            return pd.Series(dtype=float)
        return df_long.loc[mask, "Mean"].astype(float)

    # Extract series
    s_rel = _get_series("Mission Reliability (R)")
    s_av  = _get_series("Availability (%)")
    s_mmt = _get_series("MMT (Repair or PM)")
    df_fail = df_long.loc[df_long["Parameter/Indicator"] == "Number of failures", ["Component", "Mean"]].copy()

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # (1) Reliability histogram
    if len(s_rel) > 0:
        ax1.hist(s_rel.dropna().values, bins=bins)
    ax1.set_title("Mission Reliability (R)")
    ax1.set_xlabel("R")
    ax1.set_ylabel("Frequency")

    # (2) Availability histogram
    if len(s_av) > 0:
        ax2.hist(s_av.dropna().values, bins=bins)
    ax2.set_title("Availability (%)")
    ax2.set_xlabel("Availability (%)")
    ax2.set_ylabel("Frequency")

    # (3) MMT histogram
    if len(s_mmt) > 0:
        ax3.hist(s_mmt.dropna().values, bins=bins)
    ax3.set_title("MMT (Repair or PM)")
    ax3.set_xlabel("MMT")
    ax3.set_ylabel("Frequency")

    # (4) Boxplot of Number of failures by Component
    if not df_fail.empty:
        df_fail = df_fail.rename(columns={"Mean": "Failures"})
        sns.boxplot(data=df_fail, x="Component", y="Failures", ax=ax4)
    ax4.set_title("Number of failures (per Component)")
    ax4.set_xlabel("Component")
    ax4.set_ylabel("Failures per Iteration")

    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()
# *****************************************************************
# *************** Parameters definition by the user ***************
# *****************************************************************

# --- Transition-parameter dictionaries ---------------------------------------
# Each dictionary maps component IDs ("C1", "C2", ...) to transition-parameter
# settings. Time units follow your model’s unit (e.g., seconds).
#
# Transition semantics:
# - "Complete": Activity completion time ~ Normal(loc, scale)
# - "Failure_tran": Aging failure via Weibull CDF with shape k=alpha and scale λ=scale
# - "Corrective Maintenance Time": Corrective repair duration ~ Uniform(low, high)
# - "Preventive Maintenance Time": Preventive replacement duration ~ Uniform(low, high)
#

# Homogeneous setting: components share the same distributions
params_per_component_equal = {
    "C1": {
        "Complete": {"loc": 28000, "scale": 3600},
        "Failure_tran": {"dist": "weibull", "alpha": 1.5, "scale": 50000},
        #"Failure_tran": {"dist": "lognormal", "mu": 10, "sigma": 0.6},
        #"Failure_tran": {"dist": "gamma", "k": 3, "theta": 20000},
        "Corrective Maintenance Time": {"low": 7200, "high": 14400},
        "Preventive Maintenance Time": {"low": 3600, "high": 7200}
    },
    "C2": {
        "Complete": {"loc": 28000, "scale": 3600},
        "Failure_tran": {"alpha": 1.5, "scale": 50000},
        #"Failure_tran": {"dist": "lognormal", "mu": 10, "sigma": 0.6},
        #"Failure_tran": {"dist": "gamma", "k": 3, "theta": 20000},
        "Corrective Maintenance Time": {"low": 7200, "high": 14400},
        "Preventive Maintenance Time": {"low": 3600, "high": 7200}
    }
}

# Two-component heterogeneous setting: different task/failure/maintenance profiles
params_per_component_different = {
    "C1": {
        "Complete": {"loc": 28000, "scale": 3600},
        "Failure_tran": {"alpha": 1.5, "scale": 50000},
        #"Failure_tran": {"dist": "lognormal", "mu": 10, "sigma": 0.6},
        #"Failure_tran": {"dist": "gamma", "k": 3, "theta": 20000},
        "Corrective Maintenance Time": {"low": 7200, "high": 14400},
        "Preventive Maintenance Time": {"low": 3600, "high": 7200}
    },
    "C2": {
        "Complete": {"loc": 84000, "scale": 3600},
        "Failure_tran": {"alpha": 4.5, "scale": 50000},
        #"Failure_tran": {"dist": "lognormal", "mu": 10, "sigma": 0.6},
        #"Failure_tran": {"dist": "gamma", "k": 3, "theta": 20000},
        "Corrective Maintenance Time": {"low": 21600, "high": 28800},
        "Preventive Maintenance Time": {"low": 10800, "high": 14400}
    }
}

# Four-component heterogeneous setting: broader mix of behaviors
params_per_component_4 = {
    "C1": {
        "Complete": {"loc": 28000, "scale": 3600},
        "Failure_tran": {"alpha": 1.5, "scale": 50000},
        #"Failure_tran": {"dist": "lognormal", "mu": 10, "sigma": 0.6},
        #"Failure_tran": {"dist": "gamma", "k": 3, "theta": 20000},
        "Corrective Maintenance Time": {"low": 7200, "high": 14400},
        "Preventive Maintenance Time": {"low": 3600, "high": 7200}
    },
    "C2": {
        "Complete": {"loc": 84000, "scale": 3600},
        "Failure_tran": {"alpha": 4.5, "scale": 50000},
        #"Failure_tran": {"dist": "lognormal", "mu": 10, "sigma": 0.6},
        #"Failure_tran": {"dist": "gamma", "k": 3, "theta": 20000},
        "Corrective Maintenance Time": {"low": 21600, "high": 28800},
        "Preventive Maintenance Time": {"low": 10800, "high": 14400}
    },
    "C3": {
        "Complete": {"loc": 48000, "scale": 3600},
        "Failure_tran": {"alpha": 2.5, "scale": 60000},
        #"Failure_tran": {"dist": "lognormal", "mu": 10, "sigma": 0.6},
        #"Failure_tran": {"dist": "gamma", "k": 3, "theta": 20000},
        "Corrective Maintenance Time": {"low": 10800, "high": 18000},
        "Preventive Maintenance Time": {"low": 5400, "high": 9000}
    },
    "C4": {
        "Complete": {"loc": 16000, "scale": 3600},
        "Failure_tran": {"alpha": 3.5, "scale": 55000},
        #"Failure_tran": {"dist": "lognormal", "mu": 10, "sigma": 0.6},
        #"Failure_tran": {"dist": "gamma", "k": 3, "theta": 20000},
        "Corrective Maintenance Time": {"low": 28800, "high": 36000},
        "Preventive Maintenance Time": {"low": 14400, "high": 21600}
    }
}

# --- Simulation parameters & Monte Carlo run --------------------
# Scenario intent:
# - Single-component simulation (num_components=1 → labels C1, ...).
# - Horizon T_MAX=200000 (model time units).
# - Parameter set: params_per_component_equal (homogeneous components).
# - num_pm_cycles=1000 schedules preventive-maintenance cycles within the horizon.
scenario_1_c = partial(
    simulate_multicolor_petri,
    T_MAX=200000,
    num_components=2,
    params_per_component=params_per_component_different,  # swap here to try *_different or *_4
    num_pm_cycles=5
)

# Monte Carlo (N=1000):
# Returns (summary, df_results, summary_avg, by_component_dict):
# - esc_1_c_desc:        per-component summary across iterations
# - esc_1_c_long_format: long-format results by iteration
# - esc_1_c_general:     overall average across components (empty if only one)
# - esc_1_c_comparacion: dict of wide tables; includes "Average" if >1 component
esc_1_c_desc, esc_1_c_long_format, esc_1_c_general, esc_1_c_comparacion = monte_carlo_indicators(
    scenario_1_c,
    n_iterations=1000
)

# Call the function (saves 'components_comparison_boxplot.pdf' and shows the figure)
plot_boxplots_per_indicator(esc_1_c_long_format)
plot_ram_summary(esc_1_c_long_format)