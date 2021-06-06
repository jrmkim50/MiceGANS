0. cd into the folder you want to use.
1. Two datasets: GrayScale and SplitData. GrayScale contains the CT images in grayscale. SplitData contains original CT images.
2. Important folders and steps

2a. Use cuda 10.0 and cudnn 7.6. I would turn on cudnn 7.6 and then turn on cuda 10.0 (because the module load command for cudnn 7.6
loads cudnn 7.6 and cuda 12.0 but we want cuda 10.0)

2b. progressivegan3d: code for progressivegan. 

To prepare dataset (I already did, but if you add new files, you need to reprepare) run in progressivegan3d:

python3 main.py prepare
    --dataset path/to/data
    --tf_record_save_dir path/to/save/tfrecords
    --dimensionality 3

So in my case, I ran python3 main.py prepare --dataset ../GrayScale --tf_record_save_dir ../GrayScale/GAN_DS --dimensionality 3

To train progressivegan3d, open opts.py and edit the start_resolution option (line 41). start_resolution determines what resolution to start training at. If you choose 4, then the progan starts training at 4x4x4. If you choose 64, you start training at 64x64x64. I chose 4 because that was the default. Also, edit the target_resolution option (line 42). This line tells the GAN what resolution is the full resolution image. Finally, edit the gpus option (like 47) with the correct GPUs.

Also, open trainer.py and uncomment lines 88 and 89 if you want to load a specific model weight. 

For additional configuration, you can also play around with config.resolution_batch_size in the parse function of opts.py. I believe this variable determines how many images to use for each batch. My original training set had 30 images so I chose to use 5 images per gpu per batch when I was at a resolution of 4.

Now, you are ready to train. To train, run:

python3 main.py train --dataset path/to/tfrecord/file --dimensionality 3
In my case, I ran python3 main.py train --dataset ../GrayScale/GAN_DS --dimensionality 3

To generate images, run:
python main.py generate --dimensionality 3

2c. pix2pix

My dataset is in dataset/petct. I used b2a direction and ran it just like you showed me.