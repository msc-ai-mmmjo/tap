# Guide to Running Scripts on the `ada` Server

For experimentation, testing, etc.

## First-timers' Set-up

1. Log into a DoC login node; `<num>` can be any of 1 to 5

    ```shell
    ssh <username>@shell<num>.doc.ic.ac.uk
    ``` 

2. SSH into the ada server, and enter your Imperial password when prompted

    ```shell
    ssh ada
    ```

3. Install pixi on `/vol/bitbucket`

    ```shell
    mkdir -p /vol/bitbucket/$USER/.pixi
    curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/vol/bitbucket/$USER/.pixi bash
    ```

    Pin to version 0.63.2 (0.64.0 has bugs)

    ```shell
    PIXI_HOME=/vol/bitbucket/$USER/.pixi /vol/bitbucket/$USER/.pixi/bin/pixi self-update --version 0.63.2
    ```

4. Add to your shell config

    ```shell
    export PIXI_HOME="/vol/bitbucket/$USER/.pixi"
    export PIXI_CACHE_DIR="/vol/bitbucket/$USER/.pixi/cache"
    export PATH="$PIXI_HOME/bin:$PATH"
    ```

    then reload
    ```shell
    source ~/.bashrc  # or source ~/.zshrc
    ```

5. Clone our repo and install dependencies

    ```shell
    cd /vol/bitbucket/$USER
    git clone git@github.com:msc-ai-mmmjo/tap.git
    cd tap
    pixi install -e cuda
    ```

6. Get model weights
  
    ```shell
    mkdir -p /vol/bitbucket/$USER/olmo2-1b-instruct-weights

    HF_HUB_ENABLE_HF_TRANSFER=1 pixi run -e cuda python -c "
    from huggingface_hub import snapshot_download
    snapshot_download(
        'allenai/OLMo-2-0425-1B-Instruct',
        local_dir='/vol/bitbucket/$USER/olmo2-1b-instruct-weights',
        ignore_patterns=['.gitattributes', 'README.md']
    )
    print('Done')
    "
    ```

7. Update the weights path (`>>` appends to the file, `>` overwrites)

    ```shell
    echo "WEIGHTS_DIR = /vol/bitbucket/$USER/olmo2-1b-instruct-weights" >> .env
    ```

8. Inspect GPU usage, and run on a free one (e.g. GPU 1)

    ```shell
    nvidia-smi
    ```

    ```shell
    CUDA_VISIBLE_DEVICES=1 pixi run -e cuda python olmo_tap/experiments/hydra_demo.py
    ```


## For those who have done the setup already

1. Log into `ada`

2. Update our repo

    ```shell
    cd /vol/bitbucket/$USER/tap
    git pull
    ```
    
    Update dependencies

    ```shell
    cd tap
    pixi install -e cuda
    ```

3. Do whatever it is you wanted to