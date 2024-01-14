# Running the files 
### Global Planners
**Run Script** <br/>
`demo_planners.py` Runs the global path planners. 
Inside this file, you can change the path planning algorithm by setting the `DEFAULT_PLANNER` variable to `"Dijkstra"`, `"RRT_star"`, or `"Bidirectional_RRT"`. 

When running the `"Dijkstra"` planner, you can change the resolution of the grid by changing the variable `SPACING` inside the file `environment/variables.py`. The smaller this spacing is, the longer it will take for the algorithm to run, but the greater the likeliness of it finding a feasible path. However, the default spacing of 0.2 should be sufficient to find a solution in all standard environments.

When running `"RRT_star"` or `"Bidirectional_RRT"`, you can change the number of iterations that the RRT algorithms execute by setting the `RRT_ITERATIONS` variable in `environment/variables.py`. Hence, when either algorithm does not manage to find a feasible path, please increase the number of iterations. However, the default number of iterations of 2500 should be more than enough to find a solution in all environments.

`DEFAULT_OBSTACLES` enables you to select whether the obstacles are plotted inside the environment. By default, this variable is set to 'True'. 

**Environments**



### Velocity Obstacles