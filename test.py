import os 
from util.scheduler import Scheduler

def main():
    S = Scheduler('default_config.ini')
    S.run()
    
if __name__=='__main__':
    main()