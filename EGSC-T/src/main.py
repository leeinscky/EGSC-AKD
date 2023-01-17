from utils import tab_printer
from egsc import EGSCTrainer
from parser import parameter_parser
import wandb

def main():
    
    args = parameter_parser()
    tab_printer(args)
    if args.wandb:
        wandb.init(config=args, project="Efficient_Graph_Similarity_Computation_EGSC-T", settings=wandb.Settings(start_method="fork"))
    trainer = EGSCTrainer(args)
    
    trainer.fit()
    trainer.score()
    trainer.save_model()
    
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


if __name__ == "__main__":
    main()
