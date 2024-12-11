python main/test.py reconstruct --device gpu:0 \
                                --dataset cifar10 \
                                --image-size 32 \
                                --num-samples 64 \
                                --save-path ./outputs/vae/reconstruction \
                                --write-mode image \
                                ./outputs/checkpoints/vae--epoch=999-train_loss=0.0000.ckpt \