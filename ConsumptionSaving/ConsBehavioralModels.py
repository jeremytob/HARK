'''
Classes to solve and simulate consumption-savings model with a discrete, exogenous,
stochastic Markov state.  The only solver here extends ConsIndShockModel to
include a Markov state; the interest factor, permanent growth factor, and income
distribution can vary with the discrete state.
'''
import sys 
sys.path.insert(0,'../')

from copy import deepcopy
import numpy as np
from ConsIndShockModel import ConsIndShockSolver, ValueFunc, MargValueFunc, ConsumerSolution, IndShockConsumerType
from HARKutilities import warnings  # Because of "patch" to warnings modules
from HARKsimulation import drawDiscrete
from HARKinterpolation import CubicInterp, LowerEnvelope, LinearInterp
from HARKutilities import approxMeanOneLognormal, addDiscreteOutcomeConstantMean,\
                          combineIndepDstns, makeGridExpMult, CRRAutility, CRRAutilityP, \
                          CRRAutilityPP, CRRAutilityP_inv, CRRAutility_invP, CRRAutility_inv, \
                          CRRAutilityP_invP
                          

                          
utility       = CRRAutility
utilityP      = CRRAutilityP
utilityPP     = CRRAutilityPP
utilityP_inv  = CRRAutilityP_inv
utility_invP  = CRRAutility_invP
utility_inv   = CRRAutility_inv
utilityP_invP = CRRAutilityP_invP

class ConsQuasiHyperbolicSolver(ConsIndShockSolver):
    '''
    A class to solve dynamic programming problems for naive quasi-hyperbolic consumers.
    Extends ConsIndShockSolver, with identical inputs.  Computes an exponential solution if necessary,
    or optionally 
    '''
    def __init__(self,solution_next,IncomeDstn,LivPrb,DiscFac,SRDiscFac,SRDiscFacE,
                 CRRA,Rfree,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
        '''
        Constructor for a new solver for a one period problem with risky income
        and transitions between discrete Markov states.  In the descriptions below,
        N is the number of discrete states.
        
        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.    
        DiscFac : float
            Intertemporal discount factor for future utility.        
        SRDiscFac : float
            Short-run intertemporal discount factor for future utility.        
        SRDiscFacE : float
            Erroneously anticipated short-run intertemporal discount factor.   
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroFac : float
            Expected permanent income growth factor at the end of this period.
        BoroCnstArt: float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.
        aXtraGrid: np.array
            Array of "extra" end-of-period asset values-- assets above the
            absolute minimum acceptable level.
        vFuncBool: boolean
            An indicator for whether the value function should be computed and
            included in the reported solution.
        CubicBool: boolean
            An indicator for whether the solver should use cubic or linear inter-
            polation.
       
                        
        Returns
        -------
        None
        '''
        # Set basic attributes of the problem

        self.assignParameters(solution_next,IncomeDstn,LivPrb,DiscFac,
                         CRRA,Rfree,SRDiscFac,SRDiscFacE,PermGroFac,
                         BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)
   
        self.defUtilityFuncs()

    def assignParameters(self,solution_next,IncomeDstn,LivPrb,DiscFac,
                         CRRA,Rfree,SRDiscFac,SRDiscFacE,PermGroFac,
                         BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
        '''
        Assigns period parameters as attributes of self for use by other methods
        
        Parameters
        ----------
        solution_next : ConsumerSolution
            The solution to next period's one period problem.
        IncomeDstn : [np.array]
            A list containing three arrays of floats, representing a discrete
            approximation to the income process between the period being solved
            and the one immediately following (in solution_next). Order: event
            probabilities, permanent shocks, transitory shocks.
        LivPrb : float
            Survival probability; likelihood of being alive at the beginning of
            the succeeding period.    
        DiscFac : float
            Intertemporal discount factor for future utility.
        SRDiscFac : float
            Short-run intertemporal discount factor for future utility.        
        SRDiscFacE : float
            Erroneously anticipated short-run intertemporal discount factor.             
        CRRA : float
            Coefficient of relative risk aversion.
        Rfree : float
            Risk free interest factor on end-of-period assets.
        PermGroFac : float
            Expected permanent income growth factor at the end of this period.
        BoroCnstArt: float or None
            Borrowing constraint for the minimum allowable assets to end the
            period with.  If it is less than the natural borrowing constraint,
            then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
            rowing constraint.
        aXtraGrid: np.array
            Array of "extra" end-of-period asset values-- assets above the
            absolute minimum acceptable level.
        vFuncBool: boolean
            An indicator for whether the value function should be computed and
            included in the reported solution.
        CubicBool: boolean
            An indicator for whether the solver should use cubic or linear inter-
            polation.
                        
        Returns
        -------
        none
        '''        
        
        ConsIndShockSolver.assignParameters(self,solution_next,IncomeDstn,LivPrb,DiscFac,CRRA,Rfree,
                                            PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool)

        self.SRDiscFacE = SRDiscFacE
        self.SRDiscFac  = SRDiscFac
           
#    def calcEndOfPrdvP(self):
#        '''
#        Calculate end-of-period marginal value of assets at each point in aNrmNow.
#        Does so by taking a weighted sum of next period marginal values across
#        income shocks (in a preconstructed grid self.mNrmNext).
#        
#        Parameters
#        ----------
#        none
#        
#        Returns
#        -------
#        EndOfPrdvP : np.array
#            A 1D array of end-of-period marginal value of assets
#        '''        
#
#        EndOfPrdvP  = self.DiscFacEff*self.Rfree*self.PermGroFac**(-self.CRRA)*np.sum(
#                      self.PermShkVals_temp**(-self.CRRA)*
#                      self.EXPvPfuncNext(self.mNrmNext)*self.ShkPrbs_temp,axis=0)  
#        return EndOfPrdvP
#        
###############################################################################
###############################################################################

def solveConsNaive(solution_next,IncomeDstn,LivPrb,DiscFac,SRDiscFac,SRDiscFacE,
                 CRRA,Rfree,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
    '''
    Solves a single period consumption-saving problem with CRRA utility and risky
    income (subject to permanent and transitory shocks).  Can generate a value
    function if requested; consumption function can be linear or cubic splines.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncomeDstn : [np.array]
        A list containing three arrays of floats, representing a discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, permanent shocks, transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.    
    DiscFac : float
        Intertemporal discount factor for future utility.        
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        Indicator for whether the solver should use cubic or linear interpolation.
        
    Returns
    -------
    solution_now : ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (using cubic or linear splines), a marginal
        value function vPfunc, a minimum acceptable level of normalized market
        resources mNrmMin, normalized human wealth hNrm, and bounding MPCs MPCmin
        and MPCmax.  It might also have a value function vFunc and marginal mar-
        ginal value function vPPfunc.
    '''

#    if solution_next.EXPvPfuncNext == None:
#        pass
        #(solve problem for exponential consumer.  self.EXPvPfuncNext = [result for vPfuncNext])
        # init a ConsIndShockSUmer Type

        # solve the COnsIndShock problem

        # assign the ConsIndShock continuation value to the QH naive consumer type

        # iterate one period for the QH consumer

    # The case of perfect naivete is very simple.
    DiscFac = DiscFac * SRDiscFac

    solver = ConsQuasiHyperbolicSolver(solution_next,IncomeDstn,LivPrb,DiscFac,
                                       SRDiscFac,SRDiscFacE,CRRA,Rfree,
                                       PermGroFac,BoroCnstArt,aXtraGrid,
                                       vFuncBool,CubicBool) 

    solver.prepareToSolve()       # Do some preparatory work
    solution_now = solver.solve() # Solve the period
    return solution_now  

def solveConsQuasiHyperbolic(solution_next,IncomeDstn,LivPrb,DiscFac,SRDiscFac,SRDiscFacE,
                 CRRA,Rfree,PermGroFac,BoroCnstArt,aXtraGrid,vFuncBool,CubicBool):
    '''
    Solves a single period consumption-saving problem with CRRA utility and risky
    income (subject to permanent and transitory shocks).  Can generate a value
    function if requested; consumption function can be linear or cubic splines.

    Parameters
    ----------
    solution_next : ConsumerSolution
        The solution to next period's one period problem.
    IncomeDstn : [np.array]
        A list containing three arrays of floats, representing a discrete
        approximation to the income process between the period being solved
        and the one immediately following (in solution_next). Order: event
        probabilities, permanent shocks, transitory shocks.
    LivPrb : float
        Survival probability; likelihood of being alive at the beginning of
        the succeeding period.    
    DiscFac : float
        Intertemporal discount factor for future utility.        
    CRRA : float
        Coefficient of relative risk aversion.
    Rfree : float
        Risk free interest factor on end-of-period assets.
    PermGroFac : float
        Expected permanent income growth factor at the end of this period.
    BoroCnstArt: float or None
        Borrowing constraint for the minimum allowable assets to end the
        period with.  If it is less than the natural borrowing constraint,
        then it is irrelevant; BoroCnstArt=None indicates no artificial bor-
        rowing constraint.
    aXtraGrid: np.array
        Array of "extra" end-of-period asset values-- assets above the
        absolute minimum acceptable level.
    vFuncBool: boolean
        An indicator for whether the value function should be computed and
        included in the reported solution.
    CubicBool: boolean
        Indicator for whether the solver should use cubic or linear interpolation.
        
    Returns
    -------
    solution_now : ConsumerSolution
        The solution to the single period consumption-saving problem.  Includes
        a consumption function cFunc (using cubic or linear splines), a marginal
        value function vPfunc, a minimum acceptable level of normalized market
        resources mNrmMin, normalized human wealth hNrm, and bounding MPCs MPCmin
        and MPCmax.  It might also have a value function vFunc and marginal mar-
        ginal value function vPPfunc.
    '''

#    if solution_next.EXPvPfuncNext == None:
#        pass
        #(solve problem for exponential consumer.  self.EXPvPfuncNext = [result for vPfuncNext])
        # init a ConsIndShockSUmer Type

        # solve the COnsIndShock problem

        # assign the ConsIndShock continuation value to the QH naive consumer type

        # iterate one period for the QH consumer

    # The general quasi-hyperbolic case requires much more work.
    raise NotImplementedError
    DiscFac = DiscFac * SRDiscFac

    solver = ConsQuasiHyperbolicSolver(solution_next,IncomeDstn,LivPrb,DiscFac,
                                       SRDiscFac,SRDiscFacE,CRRA,Rfree,
                                       PermGroFac,BoroCnstArt,aXtraGrid,
                                       vFuncBool,CubicBool) 

    solver.prepareToSolve()       # Do some preparatory work
    solution_now = solver.solve() # Solve the period
    return solution_now  



###############################################################################       
###############################################################################

class ConsQuasiHyperbolicType(IndShockConsumerType):
    '''
    A consumer type with idiosyncratic shocks to permanent and transitory income.
    His problem is defined by a sequence of income distributions, survival prob-
    abilities, and permanent income growth rates, as well as time invariant values
    for risk aversion, discount factor, the interest rate, the grid of end-of-
    period assets, and an artificial borrowing constraint.
    '''        
    time_inv_ = IndShockConsumerType.time_inv_ + ['SRDiscFacE','SRDiscFac']


    def __init__(self,cycles=0,time_flow=True,**kwds):
        '''
        Instantiate a new ConsumerType with given data.
        See ConsumerParameters.init_idiosyncratic_shocks for a dictionary of
        the keywords that should be passed to the constructor.
        
        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.
        time_flow : boolean
            Whether time is currently "flowing" forward for this instance.
        
        Returns
        -------
        None
        '''       
        # Initialize a basic AgentType
        IndShockConsumerType.__init__(self,cycles=cycles,time_flow=time_flow,**kwds)

        # Add consumer-type specific objects, copying to create independent versions
        if self.SRDiscFacE==1:
            self.solveOnePeriod = solveConsNaive
        else:
            self.solveOnePeriod = solveConsQuasiHyperbolic

        self.update() # Make assets grid, income process, terminal solution



###############################################################################



if __name__ == '__main__':
    
    import ConsumerParameters as Params
    from HARKutilities import plotFuncs
    from time import clock
    from copy import copy
    mystr = lambda number : "{:.4f}".format(number)

    do_simulation           = False
    
    # Make and solve an example consumer with idiosyncratic income shocks
    IndShockExample = IndShockConsumerType(**Params.init_idiosyncratic_shocks)
    IndShockExample.cycles = 0 # Make this type have an infinite horizon
    
    start_time = clock()
    IndShockExample.solve()
    end_time = clock()
    print('Solving a consumer with idiosyncratic shocks took ' + mystr(end_time-start_time) + ' seconds.')
    IndShockExample.unpackcFunc()
    IndShockExample.timeFwd()
    
    # Plot the consumption function and MPC for the infinite horizon consumer
    print('Concave consumption function:')
    plotFuncs(IndShockExample.cFunc[0],IndShockExample.solution[0].mNrmMin,5)
#    print('Marginal consumption function:')
#    plotFuncsDer(IndShockExample.cFunc[0],IndShockExample.solution[0].mNrmMin,5)   

