import os 
from compiler import Compiler

def main():
    c = Compiler('default_config.ini')
    c.compile()
    c.model.summary()
    history = c.fit()
    c.plot_history()
    c.plot_prediction()
    
if __name__=='__main__':
    main()