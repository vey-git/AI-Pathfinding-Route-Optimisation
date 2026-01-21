<div align="center">
<img src="https://img.shields.io/badge/AI-Pathfinding-blueviolet?style=for-the-badge" /> <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/Oxford_Brookes-Year_2-CC0000?style=for-the-badge" />
<br><br>
🚀 AI Pathfinding Engine
🗺️ Intelligent Route Optimization on Real-World Road Networks



Implementing and benchmarking 5 search algorithms with 2.8x performance gains

Overview - Algorithms - Performance - Tech Stack

🎯 About
Year 2 AI coursework exploring pathfinding optimization on 3,393 nodes and 7,547 edges from Oxfordshire road networks. Implements both classical and cutting-edge search algorithms with real-world constraints.
​


📍 Start: Radcliffe Camera, Oxford
📍 End: Bicester Village
🎯 Goal: Find optimal path with minimal search
🧠 Algorithms
<table> <tr> <td width="50%">
🌟 Informed Search

A* with Euclidean Heuristic

✅ Optimal shortest path: 20.73 km
​

🔍 13,623 nodes explored

⚡ Admissible heuristic for non-grid networks
​

Bidirectional MM

🚀 2.8x faster than A*
​

🎯 Only 787 nodes explored

📄 Based on research (Holte et al., 2017)
​

A* with Logic Constraints

🚫 Avoids highways & dangerous zones
​

🧠 Uses propositional logic
​

🛣️ Real-world navigation modeling

</td> <td width="50%">
📊 Uninformed Search

Breadth-First Search (BFS)

📏 Fewest edges: 98 steps
​

⚠️ High cost: 25,385 nodes

💾 Memory intensive: 3.07 MB
​

Depth-First Search (DFS)

⚡ Fastest: 0.0024s execution
​

❌ Suboptimal: 51.44 km path
​

🔄 Explores only 900 nodes

</td> </tr> </table>
📈 Performance
<div align="center">
🏆 Algorithm Showdown: Radcliffe Camera → Bicester Village

🤖 Algorithm	⏱️ Time	🔍 Nodes	📏 Distance	💾 Memory	🎖️ Winner
A*	0.23s	13,623	20.73 km	0.77 MB	🥇 Best Path
Bidirectional MM	0.01s	787	4.88 km	0.24 MB	🥇 Most Efficient
BFS	0.07s	25,385	25.92 km	3.07 MB	🥉 Fewest Edges
DFS	0.002s	900	51.44 km	0.28 MB	🥇 Fastest
</div>
​

💡 Key Insight: Bidirectional search reduces exploration from radius d to d/2, achieving exponential improvement
​

🛠️ Tech Stack
<p align="center"> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/OSMnx-34A853?style=for-the-badge&logo=openstreetmap&logoColor=white" /> <img src="https://img.shields.io/badge/NetworkX-FF6F00?style=for-the-badge" /> <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge" /> </p>
bash
pip install osmnx networkx matplotlib
💻 Quick Start
python
# 🎯 A* Search - Optimal path
path, explored = a_star(graph, start, end)

# 🚀 Bidirectional MM - 2.8x faster
path, nodes = bidirectionalMM(graph, start, end)

# 🚫 Logic Constraints - Avoid highways
path, explored = a_star_with_logic(graph, start, end)
🔥 Highlights
<div align="center">
Feature	Impact
🎯 Euclidean Heuristic	Optimal for non-grid roads 
​
🚀 Bidirectional MM	64% fewer nodes explored 
​
🧠 Logic Integration	Real-world constraint modeling 
​
📊 Comprehensive Metrics	Time, space, path quality analysis 
​
</div>
📚 Research
Based on Holte et al. (2017) - MM: A bidirectional search algorithm that is guaranteed to meet in the middle
​

<div align="center">
COMP5045 - Introduction to AI | Oxford Brookes University | Year 2 | October 2025


</div>
