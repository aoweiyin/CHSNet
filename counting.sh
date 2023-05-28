# train VGG16Trans model on FSC
python counting.py --tag testing --no-wandb --device 0 --scheduler step --step 400 --dcsize 8 --batch-size 8 --lr 4e-5 --val-start 50 --val-epoch 10 --resume ./checkpoint/0527_fsc-baseline/100_ckpt.tar
