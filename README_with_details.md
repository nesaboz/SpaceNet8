# SpaceNet8 


## Remote machine setup

To be able to SSH into a VM I use paperspace Core machine that has ML-out-of-the-box, it has `nvidia-docker` installed. I've set up three machines with static IPs, my `/.ssh/config` file:

```text
Host Spacenet-P6000
  Hostname 184.105.86.246
  User paperspace
Host Spacenet-GPU+
  Hostname 184.105.175.58
  User paperspace
Host Spacenet-CPU
  Hostname 74.82.30.185
  User paperspace
```

Every instance had to have shared drive set up (see next section). I decided to dedicate folder called `share` for it. You will have to **re-mount the shared drive** after the instance is restarted (it's fast). Mount a shared drive to `share` folder via:
```zsh
sudo mount ~/share  # same as: sudo mount share if in home directory already
```
Please be patient, mounting might take up to a minute.

To run a docker container for baseline:
```zsh
sudo nvidia-docker run -v ~/share:/tmp/share --ipc=host -it --rm sn8/baseline:1.0 bash
```
this mounts a `share` drive to `/tmp/share`.

Shared drive currently has folders: 
- data (with SpaceNet8 data)
- repos (with users repos)
- runs (with all meta data and models)
## Set up a shared drive

This has already done and is here for reference.
### Create a shared drive

Shared drive is an excelent way to share drive accross machines. 

Machines do have to be on the same **private network**, which is created from a network tab on Paperspace Core.

Once private network is created **instances MUST be assigned to that private network, this is not a default**. 

Note there **might be a delay** for all of this to set up (likely <10 minutes) to be able to ssh into a machine.

Once you ssh, you will need several pieces of information, follow [this](https://docs.paperspace.com/core/compute/how-to/mounting-shared-drives/#linux), these are the commands you will be using:

I've used SHARE_FOLDERNAME as `/home/paperspace/share`:

```zsh
sudo vim /etc/fstab  # use any editor of your choce btw, add the following line at the end: 
# //YOUR_SHARED_DRIVE_IP/YOURSHARE   SHARE_FOLDERNAME  cifs  user=USERNAME,password=PASSWORD,rw,uid=1000,gid=1000,users 0 0
mkdir SHARE_FOLDERNAME  # if doesn't exist
sudo chown paperspace:paperspace SHARE_FOLDERNAME  # this is needed ONLY if SHARE_FOLDERNAME is outside your home directory 
sudo apt-get update
sudo apt install cifs-utils
sudo mount SHARE_FOLDERNAME  # MUST use sudo here!
df
```

### Download SpaceNet8 repo and build docker image

My home folder: **~/share/SpaceNet8** (`/home/paperspace/` is a home directory `~`).

On paperspace machine get our SpaceNet github repo [instructions](https://github.com/nesaboz/SpaceNet8.git):
```
git clone git@github.com:nesaboz/SpaceNet8.git
```


Build docker image (will take a few minutes):
```
sudo nvidia-docker build -t sn8/baseline:1.0 ~/share/SpaceNet8/docker 
```
There is a way to avoid constant `sudo` but requires messing with some json config files. For now just use `sudo`.

### Download SpaceNet8 data

Let this be folder (create via mkdir): **~/share/data**

Install `awscli`:
```zsh
sudo apt install awscli
# pip install awscli
```

Download the dataset (links also [here](https://spacenet.ai/sn8-challenge/)). Try to download data first and if needed set up aws credentials:
Log into [AWS management console](https://aws.amazon.com/console/), under "Account/Security credentials/
Create access key" get ACCESS_KEY and SECRET_KEY:
```
aws configure set aws_access_key_id ACCESS_KEY  
aws configure set aws_secret_access_key SECRET_KEY
```

Download training data and testing data:
```
aws s3 cp s3://spacenet-dataset/spacenet/SN8_floods/tarballs/Germany_Training_Public.tar.gz . ; aws s3 cp s3://spacenet-dataset/spacenet/SN8_floods/tarballs/Louisiana-East_Training_Public.tar.gz . ; aws s3 cp s3://spacenet-dataset/spacenet/SN8_floods/tarballs/Louisiana-West_Test_Public.tar.gz . 
```

Make 3 directories:
```zsh
mkdir Germany_Training_Public; mkdir Louisiana-East_Training_Public; mkdir Louisiana-West_Test_Public
```

Unzip the data, **make sure they all have their own directory**:
```
tar -xf Germany_Training_Public.tar.gz -C ./Germany_Training_Public; tar -xf Louisiana-East_Training_Public.tar.gz -C ./Louisiana-East_Training_Public; tar -xf Louisiana-West_Test_Public.tar.gz -C ./Louisiana-West_Test_Public
```

Delete the tar.gz files since they are no longer needed.

## Run docker container

Make sure remote machine in VSCode has open folder, doing this later might destroy the existing terminals and reset docker container.

Let's run the container and mount the three folders that we created in previous steps:
```
sudo nvidia-docker run -v ~/share:/tmp/share --ipc=host -it --rm sn8/baseline:1.0 bash
```

I added `--ipc=host` to the command to avoid shared memory [issue](https://github.com/pytorch/pytorch#docker-image). 

in general, option `-v /host/path:/container/path` mounts a folder, more options [here](https://docs.docker.com/engine/reference/commandline/run/).

the prompt should now look like this `root@<container_id>:/#` and the folders will be mounted in the `/tmp` folder. Rename this terminal window to **Do not delete** and don't delete it since this shuts down the container.

To attach to container from VSCode, install "Remote Development" extension [pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack), then one can attach the container in VSCode by going to a command pallette (Cmd+Shift+P) and typing "Attach to running container".

To see running containers' info from paperspace machine use:
```
docker ps
```
To stop all the docker containers:
```
docker container stop ID_or_NAME
docker container stop $(docker container ls -aq)
```

## Data Preparation
First we need to create intermediary data (this will create data/Germany_Training_Public/annotations/prepped_cleaned folder with all sorts of files, see README.md for details):
```
python baseline/data_prep/geojson_prep.py --root_dir /tmp/share/data --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public
```
Next create masks (this might 5 min):
```
python baseline/data_prep/create_masks.py --root_dir /tmp/share/data --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public
```
Let's create a split:

```
python baseline/data_prep/generate_train_val_test_csvs.py --root_dir /tmp/share/data --aoi_dirs Germany_Training_Public Louisiana-East_Training_Public --out_csv_basename sn8_data --val_percent 0.15 --out_dir /tmp/share/runs
```
## Train/validate Foundation Feature Network



Now we can train the Foundation network:
```
python baseline/train_foundation_features.py --train_csv /tmp/share/runs/sn8_data_train.csv --val_csv /tmp/share/runs/sn8_data_val.csv --save_dir /tmp/share/runs --model_name resnet34 --lr 0.0001 --batch_size 1 --n_epochs 1 --gpu 0
```
I've been running into memory issues and had to reduce the batch size to 1.
### Inference with Foundation Features Network

Write prediction tiffs to be used for postprocessing and generating the submission.csv
```
python baseline/foundation_eval.py --model_path /tmp/runs/resnet34_lr1.00e-04_bs1_03-05-2023-22-43/best_model.pth --in_csv /tmp/runs/split/sn8_data_val.csv --save_preds_dir /tmp/runs/foundation --gpu 0 --model_name resnet34
```

Write prediction .pngs for visual inspection of predictions:
```
python baseline/foundation_eval.py --model_path /tmp/runs/resnet34_lr1.00e-04_bs1_03-05-2023-22-43/best_model.pth --in_csv /tmp/runs/split/sn8_data_val.csv --save_fig_dir /path/to/output/foundation/pngs --gpu 0 --model_name resnet34
```

