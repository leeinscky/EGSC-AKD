from utils import tab_printer
from egsc_kd import EGSC_KD_Trainer
from parser import parameter_parser
import wandb
import torch
import numpy as np
import random

def main():
    args = parameter_parser()
    tab_printer(args)
    if args.wandb:
        wandb.init(config=args, project="Efficient_Graph_Similarity_Computation_EGSC-KD", settings=wandb.Settings(start_method="fork"))
    
    setup_seed(20)
    
    trainer = EGSC_KD_Trainer(args)
    trainer.load_model()
    trainer.fit()
    trainer.score()
    
    if args.notify:
        import os
        import sys
        if sys.platform == 'linux':
            os.system('notify-send EGSC "Program is finished."')
        elif sys.platform == 'posix':
            os.system("""
                      osascript -e 'display notification "EGSC" with title "Program is finished."'
                      """)
        else:
            raise NotImplementedError('No notification support for this OS.')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()
