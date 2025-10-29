# Bikini-Bottom-Pirate-Simulation

## Method review
We study maritime piracy risk in a constrained shipping corridor using an agent-based simulation model implemented in Mesa (a Python framework for agent-based modeling). The model is inspired by prior work on agent-based maritime traffic and piracy in the Western Indian Ocean, which studies how merchant vessels, pirate skiffs, and naval patrols interact in contested waters. In particular, previous research has shown that piracy risk is not spatially uniform: instead, it tends to concentrate in specific high-risk zones (e.g. chokepoints near trade routes), and it depends on how effectively naval forces can monitor and respond to distress calls from targeted ships.
In our model, we reproduce these core mechanisms at a simplified but interpretable level. The simulated environment is a continuous 2D sea region containing (i) one or more “high-risk zones” where pirates tend to patrol and launch attacks, and (ii) an optional “transit corridor,” a recommended shipping lane through which merchant vessels are encouraged (or forced) to travel so that naval escorts can cover them more efficiently.
There are three agent types:

* Merchant ships (blue): travel from origin to destination following either the corridor or a direct route. Each ship has a cruising speed and a vigilance/alertness level.

* Pirate boats (red): patrol within high-risk areas, detect passing merchant ships, chase them, and attempt to board/hijack.

* Naval assets (green): patrol along or near the transit corridor, and if they receive a distress call from an attacked ship within their response radius, they attempt to intervene and rescue.
  
The interaction rules include pursuit, attack, distress signaling, naval response, and possible rescue. The outcome of each encounter is probabilistic and depends on:
(a) merchant ship speed, (b) vigilance/crew alertness, (c) distance to the nearest navy asset at the moment of attack, and (d) whether the ship is traveling inside a naval-protected corridor.
This agent-based approach allows us to generate synthetic “voyage outcomes” under different security policies (e.g. with/without the corridor, stronger/weaker naval presence). We can then compare scenario-level statistics — such as attack attempt rate, hijack success rate, and safe arrival rate — in a controlled way.


