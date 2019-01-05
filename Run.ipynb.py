{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RLalgs is a package containing Reinforcement Learning algorithms Epsilon-Greedy, Policy Iteration, Value Iteration, Q-Learning, and SARSA.\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "from RLalgs.utils import epsilon_greedy\n",
    "import random\n",
    "\n",
    "def QLearning(env, num_episodes, gamma, lr, e):\n",
    "    \"\"\"\n",
    "    Implement the Q-learning algorithm following the epsilon-greedy exploration.\n",
    "\n",
    "    Inputs:\n",
    "    env: OpenAI Gym environment \n",
    "            env.P: dictionary\n",
    "                    P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)\n",
    "                    probability: float\n",
    "                    nextstate: int\n",
    "                    reward: float\n",
    "                    terminal: boolean\n",
    "            env.nS: int\n",
    "                    number of states\n",
    "            env.nA: int\n",
    "                    number of actions\n",
    "    num_episodes: int\n",
    "            Number of episodes of training\n",
    "    gamma: float\n",
    "            Discount factor.\n",
    "    lr: float\n",
    "            Learning rate.\n",
    "    e: float\n",
    "            Epsilon value used in the epsilon-greedy method.\n",
    "\n",
    "    Outputs:\n",
    "    Q: numpy.ndarray\n",
    "    \"\"\"\n",
    "\n",
    "    Q = np.zeros((env.nS, env.nA))\n",
    "    \n",
    "    #TIPS: Call function epsilon_greedy without setting the seed\n",
    "    #      Choose the first state of each episode randomly for exploration.\n",
    "    ############################\n",
    "    # YOUR CODE STARTS HERE\n",
    "    env.isd = [1 / env.nS] * env.nS\n",
    "    for i in num_episodes:\n",
    "        s=env.reset()\n",
    "        terminal=s in [5,7,11,12,15]\n",
    "        while not terminal:            \n",
    "            a=epsilon_greedy(Q[s],e)\n",
    "            out = env.step(a)\n",
    "            next_s=out[0]\n",
    "            r=out[1]\n",
    "            terminal=out[2]\n",
    "            Q[s,a]=Q[s,a]+lr*(r+gammma*max(Q[next_s])-Q[s,a])\n",
    "            s=next_s\n",
    "        \n",
    "\n",
    "    # YOUR CODE ENDS HERE\n",
    "    ############################\n",
    "\n",
    "    return Q"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
