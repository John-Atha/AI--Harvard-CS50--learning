# Harvard CS50's Introduction to Artificial Intelligence with Python 2021 course

### Project 4b - Nim

An AI to play the famous `Nim` game

#### Implementation
* The goal was to implement the methods `get_q_value`, `update_q_value`, `best_future_reward`, `choose_action`, of the `NimAI` class.
* We use the `Q-learning` method and make the AI train by playing the game against itself.
* We encode each `action` to have an `immediate reward`
* Each non-terminal action, also has a `future reward`
* Each `state` is represented as a tuple (x0, x1, x2, x3) where each value expresses the number of objects left in each pile
* Each action is represented as a tuple (i, j), meaning that we remove j objects from pile i
* We use the `Q-learning formula`: 
    * Q(s, a) <- Q(s, a) + alpha * (new value estimate - old value estimate) <br>
    * The `α factor` determines how much each Q-value is affected at each new step
* At each step we pick (with some `probability ε`) either the action with the highest Q-value, or a totally random action
* So, by adjusting the Q-values of each step-action pair according to their following rewards, `the machine learns` to take the most likely successfull actions each time  

- - -

* Developer: Giannis Athanasiou
* Github Username: John-Atha
* Email: giannisj3@gmail.com
