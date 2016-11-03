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

## Import the default parameter values
import ConsumerParameters as Params

## Now, create an instance of the consumer type using the default parameter values
## We create the instance of the consumer type by calling IndShockConsumerType()
## We use the default parameter values by passing **Params.init_idiosyncratic_shocks as an argument
BaselineExample = IndShockConsumerType(**Params.init_idiosyncratic_shocksE)


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
BaselineExample.Rfree       = 1.02 #change the risk-free interest rate
BaselineExample.CRRA        = 2.   # change  the coefficient of relative risk aversion
BaselineExample.BoroCnstArt = 0    # change the artificial borrowing constraint
BaselineExample.DiscFac     = .97   
BaselineExample.SRDiscFac    = 1   #chosen so that target debt-to-permanent-income_ratio is about .1
BaselineExample.SRDiscFacE     = 1   #chosen so that target debt-to-permanent-income_ratio is about .1


####################################################################################################
####################################################################################################

"""
Now, create another consumer to compare the BaselineExample to.
"""
# The easiest way to begin creating the comparison example is to just copy the baseline example.
# We can change the parameters we want to change later.
from copy import deepcopy
PerfectNaivete = deepcopy(BaselineExample)
PerfectNaivete.SRDiscFac     = .5
PerfectNaivete.SRDiscFacE    = 1 

"""
For both types, abstract away from survival probabilities.
"""

PerfectNaivete.LivPrb = 1
BaselineExample.LivPrb = 1

####################################################################################################
"""
Now we are ready to solve the consumers' problems.
In HARK, this is done by calling the solve() method of the ConsumerType.
"""

### First solve the baseline example.
BaselineExample.solve()

### Now solve the comparison example of the consumer with a bit more credit
PerfectNaivete.solve()



####################################################################################################
"""
Now that we have the solutions to the 2 different problems, we can compare them
"""

## We are going to compare the consumption functions for the two different consumers.
## Policy functions (including consumption functions) in HARK are stored as attributes
## of the *solution* of the ConsumerType.  The solution, in turn, is a list, indexed by the time
## period the solution is for.  Since in this demo we are working with infinite-horizon models
## where every period is the same, there is only one time period and hence only one solution.
## e.g. BaselineExample.solution[0] is the solution for the BaselineExample.  If BaselineExample
## had 10 time periods, we could access the 5th with BaselineExample.solution[4] (remember, Python
## counts from 0!)  Therefore, the consumption function cFunc from the solution to the
## BaselineExample is BaselineExample.solution[0].cFunc


## First, declare useful functions to plot later

def FirstDiffMPC_Income(x):
    # Approximate the MPC out of income by giving the agent a tiny bit more income,
    # and plotting the proportion of the change that is reflected in increased consumption

    # First, declare how much we want to increase income by
    # Change income by the same amount we change credit, so that the two MPC
    # approximations are comparable
    income_change = credit_change

    # Now, calculate the approximate MPC out of income
    return (BaselineExample.solution[0].cFunc(x + income_change) - 
            BaselineExample.solution[0].cFunc(x)) / income_change


def FirstDiffMPC_Credit(x):
    # Approximate the MPC out of credit by plotting how much more of the increased credit the agent
    # with higher credit spends
    return (PerfectNaivete.solution[0].cFunc(x) - 
            BaselineExample.solution[0].cFunc(x)) / credit_change 



## Now, plot the functions we want

# Import a useful plotting function from HARKutilities
from HARKutilities import plotFuncs
import pylab as plt # We need this module to change the y-axis on the graphs


# Declare the upper limit for the graph
x_max = 10.


# Note that plotFuncs takes four arguments: (1) a list of the arguments to plot, 
# (2) the lower bound for the plots, (3) the upper bound for the plots, and (4) keywords to pass
# to the legend for the plot.

# Plot the consumption functions to compare them
print('Consumption functions:')
plotFuncs([BaselineExample.solution[0].cFunc,PerfectNaivete.solution[0].cFunc],
           BaselineExample.solution[0].mNrmMin,x_max,
           legend_kwds = {'loc': 'upper left', 'labels': ["Baseline","XtraCredit"]})


# Plot the MPCs to compare them
print('MPC out of Credit v MPC out of Income')
plt.ylim([0.,1.2])
plotFuncs([FirstDiffMPC_Credit,FirstDiffMPC_Income],
          BaselineExample.solution[0].mNrmMin,x_max,
          legend_kwds = {'labels': ["MPC out of Credit","MPC out of Income"]})

