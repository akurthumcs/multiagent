# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

def nearestFood(food, pos):
        min = 10000

        for x in range(food.width):
            for y in range(food.height):
                if food[x][y]:
                    distance = manhattanDistance(pos, (x, y))
                    if distance < min:
                        min = distance

        return min

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]
                 
    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"   
        #Calculate distance to the nearest food
        d = nearestFood(newFood, newPos)
    
        #Calculate if there is food immediately adjacent to pacman(reward +5)
        x, y = newPos
        
        isFood = oldFood[x][y]
        
        foodScore = 0
        if isFood:
            foodScore = 5

        #Calculate if there is a ghost in the next position(-10 reward if there is)
        ghostPos = successorGameState.getGhostPosition(1)
        x2, y2 = ghostPos
        ghostScore = 0
        if x == x2 and y == y2:
            ghostScore = -10

        return (1/d) + foodScore + ghostScore

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #Psuedocode from AI: A Modern Approach, Third Edition, (FIGURE 5.3)
        """function MINIMAX-DECISION(state) returns an action
                return arg max_(a in ACTIONS(s) MIN-VALUE(RESULT(state, a))
                
            function MAX_VALUE(state) returns a utility value
                if TERMINAL-TEST(state) then return UTILITY(state)
                v = -inf
                for each a in ACTIONS(state) do
                    v = MAX(v, MIN-VALUE(RESULT(s, a)))
                return v

            function MIN-VALUE(state) returns a utility value
                if TERMINAL-TEST(state) then return UTILITY(state)
                v = inf
                for each a in ACTIONS(state) do
                    v = MIN(v, MAX-VALUE(RESULT(s,a)))
                """
        #Result(s,a)
        numAgents = gameState.getNumAgents()

        def maxValue(state: gameState, depth):
            
            bestAction = None
            
            # if TERMINAL-TEST(state) then return UTILITY(state)
            if state.isWin() or state.isLose():
                return (self.evaluationFunction(state), bestAction)
            if depth == self.depth:
                return (self.evaluationFunction(state), bestAction)
            
            #v = -inf
            v = -1000000
            
            #for each a in ACTIONS(state) do
            legalActions = state.getLegalActions(0)
            if len(legalActions) == 0:
                return (self.evaluationFunction(state), None)
            for a in legalActions:
                
                #v = MAX(v, MIN-VALUE(RESULT(s, a)))
                nextv = minValue(state.generateSuccessor(0, a), 1, depth)
                if nextv[0] > v:
                    v = nextv[0]
                    bestAction = a
            
            #return v
            return (v, bestAction)
        
        def minValue(state: gameState, agentNum, depth):
        #if TERMINAL-TEST(state) then return UTILITY(state)
            bestAction = None
            
            #v = inf
            v = 1000000

            #for each a in ACTIONS(state) do
            legalActions = state.getLegalActions(agentNum)
            if len(legalActions) == 0:
                return (self.evaluationFunction(state), bestAction)
            for a in legalActions:
                if agentNum + 1 == numAgents:
                    nextv = maxValue(state.generateSuccessor(agentNum, a), depth + 1)
                    if nextv[0] < v:
                        v = nextv[0]
                        bestAction = a
        
                #v = MIN(v, MAX-VALUE(RESULT(s,a)))
                else:
                    nextv = minValue(state.generateSuccessor(agentNum, a), agentNum + 1, depth)
                    if nextv[0] < v:
                        v = nextv[0]
                        bestAction = a
                
            #return v
            return (v, bestAction)
        
        return maxValue(gameState, 0)[1]
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def maxValue(state: gameState, depth, alpha, beta):
            
            bestAction = None
            
            # if TERMINAL-TEST(state) then return UTILITY(state)
            if state.isWin() or state.isLose():
                return (self.evaluationFunction(state), bestAction)
            if depth == self.depth:
                return (self.evaluationFunction(state), bestAction)
            
            #v = -inf
            v = -1000000
            
            #for each a in ACTIONS(state) do
            legalActions = state.getLegalActions(0)
            if len(legalActions) == 0:
                return (self.evaluationFunction(state), None)
            for a in legalActions:
                
                #v = MAX(v, MIN-VALUE(RESULT(s, a)))
                nextv = minValue(state.generateSuccessor(0, a), 1, depth, alpha, beta)
                if nextv[0] > v:
                    v = nextv[0]
                    bestAction = a
                
                if v > beta:
                    return (v, bestAction)
                
                alpha = max(alpha, v)
            
            #return v
            return (v, bestAction)
        
        def minValue(state: gameState, agentNum, depth, alpha, beta):
        #if TERMINAL-TEST(state) then return UTILITY(state)
            bestAction = None
            
            #v = inf
            v = 1000000

            #for each a in ACTIONS(state) do
            legalActions = state.getLegalActions(agentNum)
            if len(legalActions) == 0:
                return (self.evaluationFunction(state), bestAction)
            for a in legalActions:
                if agentNum + 1 == numAgents:
                    nextv = maxValue(state.generateSuccessor(agentNum, a), depth + 1, alpha, beta)
                    if nextv[0] < v:
                        v = nextv[0]
                        bestAction = a
        
                #v = MIN(v, MAX-VALUE(RESULT(s,a)))
                else:
                    nextv = minValue(state.generateSuccessor(agentNum, a), agentNum + 1, depth, alpha, beta)
                    if nextv[0] < v:
                        v = nextv[0]
                        bestAction = a

                if v < alpha:
                    return (v, bestAction)
                
                beta = min(beta, v)
                
            #return v
            return (v, bestAction)
        
        return maxValue(gameState, 0, -1000000, 1000000)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def maxValue(state: gameState, depth):
            
            bestAction = None
            
            # if TERMINAL-TEST(state) then return UTILITY(state)
            if state.isWin() or state.isLose():
                return (self.evaluationFunction(state), bestAction)
            if depth == self.depth:
                return (self.evaluationFunction(state), bestAction)
            
            #v = -inf
            v = -1000000
            
            #for each a in ACTIONS(state) do
            legalActions = state.getLegalActions(0)
            if len(legalActions) == 0:
                return (self.evaluationFunction(state), bestAction)
            for a in legalActions:
                
                #v = MAX(v, MIN-VALUE(RESULT(s, a)))
                nextv = expValue(state.generateSuccessor(0, a), 1, depth)
                if nextv > v:
                    v = nextv
                    bestAction = a
            
            #return v
            return (v, bestAction)
        
        def expValue(state: gameState, agentNum, depth):
            #Initalize v = 0
            v = 0

            #for each a in ACTIONS(state) do
            legalActions = state.getLegalActions(agentNum)
            numActions = len(legalActions)
            if numActions == 0:
                return self.evaluationFunction(state)
            #Calculate probability of successor(uniform)
            p = (1 / numActions)
            for a in legalActions:
                #v+= p * value(successor)
                if agentNum + 1 == numAgents:
                    v += p * maxValue(state.generateSuccessor(agentNum, a), depth + 1)[0]
                #Generalization, more min agents
                else:
                    v += p * expValue(state.generateSuccessor(agentNum, a), agentNum + 1, depth)
                
                
            #return v
            return v
        
        return maxValue(gameState, 0)[1]



def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    "*** YOUR CODE HERE ***"   
    #Calculate distance to the nearest food
    d = nearestFood(food, pos)
    


    return (1/d) + currentGameState.getScore()


# Abbreviation
better = betterEvaluationFunction
