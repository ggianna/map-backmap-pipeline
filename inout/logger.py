import logging
import torch
import inspect

# Logging settings
class Logger(logging.getLoggerClass()):
    """Set up logging"""
    def __init__(self, name = __name__, 
                 level = logging.DEBUG, 
                 filename = "standard.log"):
        super().__init__(name)
        
        #self.setLevel(level)

        if level == logging.DEBUG:
            torch.utils.backcompat.broadcast_warning.enabled=True
            torch.autograd.set_detect_anomaly(True)

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(filename),
                logging.StreamHandler()
            ]
        )

    def info(self, msg):
        logger = logging.getLogger(self.name)
        logger.info(msg)

    def write_parameters(self, parameters):
        """Write the attributes contained in the 
        parameters dictionary in the log file"""
        logger = logging.getLogger(self.name)
        for attribute in inspect.getmembers(parameters):
            if not attribute[0].startswith('_'):  # exclude attributes that do not come from the input file
                if not attribute[0].startswith('len'): # exclude the len attribute, if present
                    logger.info( attribute[0] + "  -  " + str(attribute[1]) )

