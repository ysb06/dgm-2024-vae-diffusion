python src/baseline/test.py sample  --device gpu:0 \
                                    --image-size 32 \
                                    --seed 42 \
                                    --num-samples 1000 \
                                    --save-path ./outputs/vae/sampling \
                                    --write-mode image \
                                    512 \
                                    ./outputs/checkpoints/vae--epoch=999-train_loss=0.0000.ckpt \