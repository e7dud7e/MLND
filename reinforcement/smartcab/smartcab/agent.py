import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import OrderedDict
import numpy as np

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor
        
        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.epsilon_c = 0.02    #Constant for epsilon decay function
        self.train_count = 0     #Counts how many training trials


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
		# If 'testing' is True, set epsilon and alpha to 0    
        self.train_count +=1
        if not testing: #decrease epsilon, but keep it non-negative
            self.epsilon = max(0,self.update_epsilon(self.epsilon))
        else: #stop learning when in testing phase
            self.epsilon = 0.
            self.alpha = 0.
        # Update additional class parameters as needed

        return None

    def update_epsilon(self,epsilon):
        
        return self.update_epsilon_optimized()
	
    def update_epsilon_default(self,epsilon):
        #use linear step to decrease epsilon to zero.
        step = -0.05 #for basic q learner
        epsilon +=step
        return epsilon
    
    def update_epsilon_optimized(self):
        return np.cos(self.epsilon_c * self.train_count)
        
    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint: 'forward', 'left', 'right'
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
												#light, oncoming, left, right
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent        
        state = (waypoint, inputs['light'], inputs['oncoming'], inputs['left'])

        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
		
        maxQ = max([q for q in self.Q[state].itervalues()])

        return maxQ

    def get_maxQ_action(self,state):
        
        #Find all actions with max Q
        #Randomly choose one best action
        best_action = None
        maxQ = self.get_maxQ(state)
        if self.epsilon > 0.0: #while still exploring:   
            best_action = self.get_maxQ_action_explore(state,maxQ)
        else: #not exploring anymore
            if maxQ > 0.0: #use optimal policy
                best_action = self.get_maxQ_action_explore(state,maxQ)
            else: #estimate optimal policy
                best_action = self.get_maxQ_action_estimate(state,maxQ)
        return best_action
    
    # Use this when best action is know through direct experience,
    # and also when still in exploration mode.
    def get_maxQ_action_explore(self,state,maxQ):
        best_action_l = list()
        for action, q in self.Q[state].iteritems():
            if q == maxQ:
                best_action_l.append(action)
        return random.choice(best_action_l)
    
    #Use estimate when maxQ is 0.0
    def get_maxQ_action_estimate(self,state,maxQ):
        best_action = None
        valid_actions = list()
        action_scores = list()
        #get all the actions that weren't penalized in training
        for action, q in self.Q[state].iteritems():
            if q == 0.0: 
                valid_actions.append(action)
        action_scores = [0 for x in valid_actions] #initialize
        
        #For each valid action,
        #Search existing Q table by varying one state feature at a time
        #check if state exists in Q table first
        for i, action in enumerate(valid_actions):
            state_l = list(state)
            for waypoint in ['forward','right','left']:
                state_l[0] = waypoint
                state_t = tuple(state_l)
                if state_t not in self.Q:
                    continue
                q = self.Q[state_t][action]
                if q < 0.0:
                    action_scores[i] -= 1
                elif q > 0.0:
                    action_scores[i] +=1
                else:
                    pass

            state_l = list(state)
            for light in ['red', 'green']:
                state_l[1] = light
                state_t = tuple(state_l)
                if state_t not in self.Q:
                    continue
                q = self.Q[state_t][action]
                if q < 0.0:
                    action_scores[i] -= 1
                elif q > 0.0:
                    action_scores[i] +=1
                else:
                    pass

            state_l = list(state)
            for oncoming in [None, 'forward','right','left']:
                state_l[2] = oncoming
                state_t = tuple(state_l)
                if state_t not in self.Q:
                    continue
                q = self.Q[state_t][action]
                if q < 0.0:
                    action_scores[i] -= 1
                elif q > 0.0:
                    action_scores[i] +=1
                else:
                    pass

            state_l = list(state)
            for left in [None, 'forward', 'right', 'left']:
                state_l[3] = left
                state_t = tuple(state_l)
                if state_t not in self.Q:
                    continue
                q = self.Q[state_t][action]
                if q < 0.0:
                    action_scores[i] -= 1
                elif q > 0.0:
                    action_scores[i] +=1
                else:
                    pass

        best_i = np.argmax(action_scores)
        best_action = valid_actions[best_i]        
        return best_action

    
    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        # Then, for each action available, set the initial Q-value to 0.0
        
        #Order the actions to try the ones listed first
        #TRY None (stop) first, because we have to stop for red lights
        #Try moving forward, because it only depends on a green light
        #Try moving right, because it only results in a violation when left traffic goes forward
        #Try moving left last, because it has most ways of getting violations (oncoming goes forward or right)
        
        if state not in self.Q:
            self.Q[state] = OrderedDict([
                (None,      0.0), #stop
                ('forward', 0.0),
                ('right',   0.0),
                ('left',    0.0),])
        return
    
    #version for optimized agent
    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        if not self.learning:
            action = random.choice(self.valid_actions)
        else:
            random_action = self.true_with_probability(self.epsilon)
            if random_action:
                action = random.choice(self.valid_actions)
            else:
                action = self.get_maxQ_action(state)    
        return action

    
    def true_with_probability(self,epsilon):
        return random.random() < epsilon

    
    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        if self.learning:
            Q_before = self.Q[state][action]
            Q_updated = Q_before*(1-self.alpha) + self.alpha*reward
            self.Q[state][action] = Q_updated
        
        return

	
    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose=True)
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning=True)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent=agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    #sim = Simulator(env=env, update_delay=.01, display=False, log_metrics=True) #default
    sim = Simulator(env=env, update_delay=.01, display=False, log_metrics=True, optimized=True) #optimized
	#sim = Simulator(env=env, update_delay=1, display=True, log_metrics=True) #for sim-no-learning log
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    #sim.run(n_test=10)
    sim.run(n_test=10, tolerance=0.93) #optimized


if __name__ == '__main__':
    run()
