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

## Statement of questions of interest
The purpose of this analysis is not simply to place merchant ships, pirates, and naval vessels on a map, but rather to answer several questions directly related to "how to reduce the risk of hijacking." This analysis is based on existing research in the western Indian Ocean and our simulation mechanism implemented in MESA.

### Can the methods in the referenced paper be reused?
To figure out, what matters is “what are the important influencing factors?”. This is the core issue we need to address, such as the corresponding speed, warning range, naval support, and route selection mentioned in the previous section.
The reference paper proposes a feasible solution: to concentrate merchant ships on a protected channel.Many practical anti-piracy operations require or recommend that merchant ships sail along "recommended transit corridors." The advantage of this is that the navy can patrol a relatively narrow channel without needing to monitor the entire sea surface. Theoretically, this increases the probability of the navy being able to respond in time. However, there are also problems: if all ships take the same route, pirates will know where to ambush, and the number of attack attempts may actually increase. Therefore, we need to find a balance between these two.
So in the simulation, we need to define different scenarios:
* Scenario A: Open passage → Most ships travel along the passage;
* Scenario B: Closed passage → Ships take their own more dispersed routes;
Then, based on the frequency of pirate attacks (attempts / ships), the final proportion of ships actually hijacked (captured / ships), and whether the spatial distribution of attacks is more concentrated near the passage, we can verify whether concentrated navigation is more "easier to protect" or "easier to be targeted".

### Can the naval influence be reasonably planned, and are there any boundary effects?
Both literature and practical experience suggest that the navy plays more of a "last line of defense": they may not necessarily prevent pirates from attempting an attack, but they can prevent an attack from ending in a successful hijacking. The more critical question for policymakers is: how many patrol ships/what response radius is needed to keep the success rate of hijackings at an acceptable level?
Therefore, during the simulation, we can adjust the number of naval personnel or the size of the perimeter. Different deployment methods, such as whether or not they are positioned around the safety passage, are also options. This can provide some insight into the effectiveness of the coordination between the navy and the safety passage.
Key evaluation metrics include the success rate of hijacking (number of successful hijackings / number of attack attempts) and the number of ships rescued by the Navy. The obtained data can be used to determine the linear relationship using the Pearson correlation coefficient, and the Spearman rank correlation can be used to verify the consistency of non-linearity.

### What role do a ship's speed and alertness play in piracy incidents?
In reality, not every ship falls within naval protection. This is where the ship's inherent qualities become crucial: ships that are fast, have early detection capabilities, and can quickly issue warnings or evasive maneuvers are often more difficult to board and control. In our simulation, each ship has two attributes: speed and vigilance, and the probability of a successful attack is a function of these two quantities.
This question aims to clarify the following:
* When no navy is nearby, are high-speed/high-alert ships significantly safer?
* When a navy is nearby, is high speed still as effective?
* Can we plot a curve from simulation data, similar to "P (hijacking) decreases with speed," to illustrate that this is a quantifiable relationship?
The significance of this problem lies in its ability to provide guidance for merchant ship crew training and speed selection strategies, while also validating our initial hypothesis regarding the influence of multiple factors.

In a nutshell, descriptive statistics and visualization analysis were used to compare the number of attacks, hijacking rates, and rescue rates under different scenarios (channel open/closed, varying naval numbers) to preliminarily identify trends. Building upon this, linear regression and logistic regression models can be constructed to quantitatively measure the marginal impact of each factor:
Scenario-level linear regression is used to estimate the combined impact of naval numbers, channel policies, and pirate attack tendencies on the overall hijacking rate;
Ship-level logistic regression is used to assess the impact of variables such as individual ship speed, alertness, whether it is sailing within the channel, and distance from the nearest naval vessel on the probability of hijacking.
These analyses theoretically reveal three main trends:
* An increase in naval numbers is significantly negatively correlated with a decrease in hijacking risk;
* The combination of "channel + navy" has a synergistic effect, significantly improving security;
* Even with limited naval protection, ship speed and alertness can still effectively reduce individual risk.
