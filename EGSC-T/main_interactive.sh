
cd ~/code/EGSC-AKD/EGSC-T && conda activate Efficient-Graph-Similarity-Computation

# epochs=1
python src/main.py --dataset=AIDS700nef --gnn-operator=gin --epochs=1 --batch-size=128 --wandb=1
python src/main.py --dataset=LINUX --gnn-operator=gin --epochs=1 --batch-size=128 --wandb=1
python src/main.py --dataset=IMDBMulti --gnn-operator=gin --epochs=1 --batch-size=128 --wandb=1
python src/main.py --dataset=ALKANE --gnn-operator=gin --epochs=1 --batch-size=128 --wandb=1


############################################AIDS700nef epochs=6000 ############################################
nohup python src/main.py --dataset=AIDS700nef --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 >/dev/null 2>&1 &

############################################LINUX epochs=6000 ############################################
nohup python src/main.py --dataset=LINUX --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 >/dev/null 2>&1 &


############################################IMDBMulti epochs=6000 ############################################
nohup python src/main.py --dataset=IMDBMulti --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 >/dev/null 2>&1 &


############################################ALKANE epochs=6000 ############################################
nohup python src/main.py --dataset=ALKANE --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 >/dev/null 2>&1 &
