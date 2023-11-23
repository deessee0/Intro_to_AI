# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    start = problem.getStartState()
    if problem.isGoalState(start):
        return []
    esplorati = set()
    stack = util.Stack()
    stack.push((start, []))

    #aggiungo i nodi non esplorati 
    #(e il percorso per raggiungerli) sullo stack
    #e poi all'insieme degli esplorati
    while not stack.isEmpty():
        current_state, path = stack.pop()
        if current_state in esplorati:
            continue
        esplorati.add(current_state)

        #se lo stato attuale è il goal restituisco il path al nodo
        if problem.isGoalState(current_state):
            return path  
        
        for successor, action, _ in problem.getSuccessors(current_state):
            if successor not in esplorati:
                #aggiungo il nodo successivo il suo path allo stack
                stack.push((successor, path + [action]))

    # se non trovo il goal
    return [] 


def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #come prima ma con la Queue
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []
    coda = util.Queue()
    esplorati = set([start]) 
    coda.push((start, []))

    while not coda.isEmpty():
        current_state, path = coda.pop()
        if problem.isGoalState(current_state):
            return path  
        for successor, action, _ in problem.getSuccessors(current_state):
            if successor not in esplorati:
                esplorati.add(successor)  
                coda.push((successor, path + [action]))

    return []

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    """
    Search the node of least total cost first using the Uniform Cost Search algorithm.
    """

    priorityQueue = util.PriorityQueue()
    start = problem.getStartState()
    priorityQueue.push((start, [], 0), 0)  # (state, path, cost), priority

    # Dizionario di nodi visititati 
    # con il costo minimo per raggiungerli
    esplorati = {}

    while not priorityQueue.isEmpty():
        #come prima ma tengo traccia del costo
        current_state, path, current_cost = priorityQueue.pop()
        if problem.isGoalState(current_state):
            return path

        # Check del costo ottimale
        if current_state not in esplorati or current_cost < esplorati[current_state]:
            esplorati[current_state] = current_cost
            
            #ciclo sui successivi come prima
            for successor, action, step_cost in problem.getSuccessors(current_state):
                new_cost = current_cost + step_cost
                new_path = path + [action]
                priorityQueue.update((successor, new_path, new_cost), new_cost)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    #come prima
    start_state = problem.getStartState()
    priorityQueue = util.PriorityQueue()
    priorityQueue.push((start_state, [], 0), 0) 
    visited = {}


    while not priorityQueue.isEmpty():
        current_state, path, current_cost = priorityQueue.pop()
        if problem.isGoalState(current_state):
            return path

        if current_state not in visited or current_cost < visited[current_state]:
            visited[current_state] = current_cost

            # Come prima ma con stima del costo con euristica
            for successor, action, step_cost in problem.getSuccessors(current_state):
                new_cost = current_cost + step_cost
                estimated_cost_to_goal = new_cost + heuristic(successor, problem)
                new_path = path + [action]
                priorityQueue.update((successor, new_path, new_cost), estimated_cost_to_goal)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
