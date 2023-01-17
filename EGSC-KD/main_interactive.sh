
cd ~/code/EGSC-AKD/EGSC-KD && conda activate Efficient-Graph-Similarity-Computation

python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=1 --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --wandb=0
python src/main_kd.py --dataset=LINUX --gnn-operator=gin --epochs=1 --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --wandb=0
python src/main_kd.py --dataset=IMDBMulti --gnn-operator=gin --epochs=1 --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --wandb=0
python src/main_kd.py --dataset=ALKANE --gnn-operator=gin --epochs=1 --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --wandb=0


############################################AIDS700nef epochs=9 ############################################
# nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=9 \
# --batch-size=128 --learning-rate=0.001 --use-adversarial=0 --adversarial_ouput_class=1 --cuda-id=3 >/dev/null 2>&1 &



############################################AIDS700nef epochs=3000 ############################################
# nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=3000 \
# --batch-size=128 --learning-rate=0.001 --use-adversarial=0 --adversarial_ouput_class=1 --cuda-id=0 >/dev/null 2>&1 &

# nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=3000 \
# --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 >/dev/null 2>&1 &

# nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=3000 \
# --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=16 >/dev/null 2>&1 &



############################################AIDS700nef epochs=6000 ############################################
nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=0 --adversarial_ouput_class=1 --cuda-id=0 --light=0 >/dev/null 2>&1 &

nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=1 --light=0 >/dev/null 2>&1 &

nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=16 --cuda-id=2 --light=0 >/dev/null 2>&1 &

# light model
nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=3 --light=1 >/dev/null 2>&1 &


############################################AIDS700nef epochs=7000 ############################################
# nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=7000 \
# --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=3 --light=0 >/dev/null 2>&1 &

############################################AIDS700nef epochs=8000 ############################################
# nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=8000 \
# --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=1 >/dev/null 2>&1 &

############################################AIDS700nef epochs=9000 ############################################
# nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=9000 \
# --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=2 >/dev/null 2>&1 &

############################################AIDS700nef epochs=10000 ############################################
# nohup python src/main_kd.py --dataset=AIDS700nef --gnn-operator=gin --epochs=10000 \
# --batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=3 >/dev/null 2>&1 &












############################################LINUX epochs=6000 ############################################
nohup python src/main_kd.py --dataset=LINUX --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=0 --adversarial_ouput_class=1 --cuda-id=0 --light=0 >/dev/null 2>&1 &

nohup python src/main_kd.py --dataset=LINUX --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=1 --light=0 >/dev/null 2>&1 &

nohup python src/main_kd.py --dataset=LINUX --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=0 --adversarial_ouput_class=1 --cuda-id=1 --light=0 >/dev/null 2>&1 &

nohup python src/main_kd.py --dataset=LINUX --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=0 --light=0 >/dev/null 2>&1 &

############################################IMDBMulti epochs=6000 ############################################
nohup python src/main_kd.py --dataset=IMDBMulti --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=0 --adversarial_ouput_class=1 --cuda-id=2 --light=0 >/dev/null 2>&1 &

nohup python src/main_kd.py --dataset=IMDBMulti --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=3 --light=0 >/dev/null 2>&1 &

nohup python src/main_kd.py --dataset=IMDBMulti --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=0 --adversarial_ouput_class=1 --cuda-id=3 --light=0 >/dev/null 2>&1 &

nohup python src/main_kd.py --dataset=IMDBMulti --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=2 --light=0 >/dev/null 2>&1 &

############################################ALKANE epochs=6000 ############################################
nohup python src/main_kd.py --dataset=ALKANE --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=0 --adversarial_ouput_class=1 --cuda-id=0 --light=0 >/dev/null 2>&1 &

nohup python src/main_kd.py --dataset=ALKANE --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=1 --light=0 >/dev/null 2>&1 &

nohup python src/main_kd.py --dataset=ALKANE --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=0 --adversarial_ouput_class=1 --cuda-id=1 --light=0 >/dev/null 2>&1 &

nohup python src/main_kd.py --dataset=ALKANE --gnn-operator=gin --epochs=6000 \
--batch-size=128 --learning-rate=0.001 --use-adversarial=1 --adversarial_ouput_class=1 --cuda-id=0 --light=0 >/dev/null 2>&1 &