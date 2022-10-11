import torch
import numpy as np

class TScheduler():
    """Scheduler for the Temperature parameter  
    used in the Gumbel distribution sampling
    """
    def __init__(self, Tstart, 
                       tmin, 
                       n_epochs, 
                       decay_ratio):

        """Scheduler initialization. Calculate vector of 
        temperature values for each epoch following
        an exponential decay until the tmin value is reached"""

        epoch = np.linspace(0, n_epochs+1, n_epochs+1) #TODO remove need for +1
        decay_epoch = int(n_epochs * decay_ratio) + 0.5 
                                # + 0.5 added to round always at least to 1  
                                # and not divide by 0 in the next line
        t_sched = Tstart * np.exp(-epoch/decay_epoch) +  tmin
        self.t_sched = torch.Tensor(t_sched)

        # Initialize the state of the Scheduler
        self.current_step = 0

    def step(self):
        """Update the state of the Scheduler"""
        self.current_step += 1
    
    def current_temperature(self):
        """Return the temperature value corresponding 
        to the current state"""
        return self.t_sched[self.current_step]