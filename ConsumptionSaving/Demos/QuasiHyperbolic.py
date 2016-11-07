"""
This file solves the perfectly naive, perfectly sophisticated, and partially naive 
quasi-hyperbolic discounting models and reports the results.
"""


####################################################################################################
####################################################################################################
"""
The first step is to create the ConsumerType we want to solve the model for.
"""

# The first step is to be able to bring things in from different directories
import sys 
import os
sys.path.insert(0, os.path.abspath('../')) #Path to ConsumptionSaving folder
sys.path.insert(0, os.path.abspath('../../'))

## Import the HARK ConsumerType we want 
## Here, we bring in an agent making a consumption/savings decision every period, subject
## to transitory and permanent income shocks.
from ConsIndShockModel import IndShockConsumerType
from ConsBehavioralModels import ConsNaiveHyperbolicType


## Import the default parameter values
import ConsumerParameters as Params

## Now, create an instance of the consumer type using the default parameter values
## We create the instance of the consumer type by calling IndShockConsumerType()
## We use the default parameter values by passing **Params.init_idiosyncratic_shocks as an argument
BaselineExample = IndShockConsumerType(**Params.init_idiosyncratic_shocks)
PerfectNaivete  = ConsNaiveHyperbolicType(**Params.init_idiosyncratic_shocksB)

####################################################################################################
####################################################################################################

"""
The next step is to change the values of parameters as we want.

To see all the parameters used in the model, along with their default values, see
ConsumerParameters.py

Parameter values are stored as attributes of the ConsumerType the values are used for.
For example, the risk-free interest rate Rfree is stored as BaselineExample.Rfree.
BaselineExample.Rfree = 1.02
"""

## Change some parameter values
BaselineExample.Rfree       = 1.001 #change the risk-free interest rate
BaselineExample.CRRA        = 1.001   # change  the coefficient of relative risk aversion
BaselineExample.BoroCnstArt = 0.    # change the artificial borrowing constraint
BaselineExample.DiscFac     = .999
BaselineExample.SRDiscFac    = 1.  
BaselineExample.SRDiscFacE     = 1.
BaselineExample.PermGroFac = [1.]
BaselineExample.PermShkStd = [0.000001]                  
BaselineExample.TranShkStd = [0.000001] 
BaselineExample.updateIncomeProcess()
BaselineExample.CubicBool = False
BaselineExample.vFuncBool = True

PerfectNaivete.Rfree       = 1.001 #change the risk-free interest rate
PerfectNaivete.CRRA        = 1.001   # change  the coefficient of relative risk aversion
PerfectNaivete.BoroCnstArt = 0.    # change the artificial borrowing constraint
PerfectNaivete.DiscFac     = .999   
PerfectNaivete.SRDiscFac    = .5   
PerfectNaivete.SRDiscFacE     = 1. 
PerfectNaivete.PermGroFac = [1.]
PerfectNaivete.PermShkStd = [0.000001]                  
PerfectNaivete.TranShkStd = [0.000001] 
PerfectNaivete.updateIncomeProcess()
PerfectNaivete.CubicBool = False

####################################################################################################
####################################################################################################

"""
Now, create another consumer to compare the BaselineExample to.
"""
# The easiest way to begin creating the comparison example is to just copy the baseline example.
# We can change the parameters we want to change later.
from copy import deepcopy
#PerfectNaivete = deepcopy(BaselineExample)
#PerfectNaivete.SRDiscFac     = .5
#PerfectNaivete.SRDiscFacE    = 1 

"""
For both types, abstract away from complications.
"""

PerfectNaivete.LivPrb = [1.]
BaselineExample.LivPrb = [1.]
BaselineExample.cycles = 0
PerfectNaivete.cycles = 1

####################################################################################################
"""
Now we are ready to solve the consumers' problems.
In HARK, this is done by calling the solve() method of the ConsumerType.
"""

### First solve the baseline example.
BaselineExample.solve()

PerfectNaivete.cFunc_terminal_ = BaselineExample.solution[0].cFunc
PerfectNaivete.vFunc_terminal_ = BaselineExample.solution[0].vFunc
#PerfectNaivete.EXPvPfuncNext = BaselineExample.vPfuncNext

### Now solve the comparison example of the consumer under the assumption of 
### perfectly naive quasi-hyperbolic discounting.
PerfectNaivete.solve()


## Now, plot the functions we want

# Import a useful plotting function from HARKutilities
from HARKutilities import plotFuncs
import pylab as plt # We need this module to change the y-axis on the graphs


# Declare the upper limit for the graph
x_max = 100.


# Note that plotFuncs takes four arguments: (1) a list of the arguments to plot, 
# (2) the lower bound for the plots, (3) the upper bound for the plots, and (4) keywords to pass
# to the legend for the plot.

# Plot the consumption functions to compare them
print('Consumption functions:')
plotFuncs([BaselineExample.solution[0].cFunc,PerfectNaivete.solution[0].cFunc],
           BaselineExample.solution[0].mNrmMin,x_max,
           legend_kwds = {'loc': 'upper left', 'labels': ["Baseline","Perfect Naivete"]})

print BaselineExample.solution[0].cFunc(range(10))
print PerfectNaivete.solution[0].cFunc(range(10))
                          
                          
## Plot the MPCs to compare them
#plt.ylim([0.,1.2])
#plotFuncs([FirstDiffMPC_Credit,FirstDiffMPC_Income],
#          BaselineExample.solution[0].mNrmMin,x_max,
#          legend_kwds = {'labels': ["MPC out of Credit","MPC out of Income"]})

