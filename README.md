# Diffusion360_ComfyUI
ComfyUI plugin of https://github.com/ArcherFMY/SD-T2I-360PanoImage

## features
- [x] text-to-panoimage pipeline
- [x] image-to-panoimage pipeline

## Installation
1. Install ComfyUI following https://github.com/comfyanonymous/ComfyUI
2. Install this plugin by the following commands.

   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/ArcherFMY/Diffusion360_ComfyUI
   cd Diffusion360_ComfyUI
   pip install -r requirements.txt
   ```
__Note that I recommend to install 0.20.0<= diffusers <= 0.26.0. The higher diffusers version will get an over-saturated SR result.__

## Usage
### prepare the models
Download models from [HF](https://huggingface.co/archerfmy0831/sd-t2i-360panoimage). Rename the folder `sd-t2i-360panoimage` to `diffusion360` and put the folder into `models/diffusers`.
```
${ComfyUI_ROOT} 
|-- models
    |-- diffusers
        |-- diffusion360
            |-- data  
            |   |-- a-living-room.png
            |   |...
            |-- models  
            |   |-- sd-base
            |   |-- sr-base
            |   |-- sr-control
            |   |-- RealESRGAN_x2plus.pth
            |-- txt2panoimg
            |-- img2panoimg
            |...
```

### load workflows
Load workflows from the folder `workflows/`
```
${This_Repo_ROOT} 
|-- workflows
    |-- Text2Pano.json ## text-to-pano pipeline
    |-- Image2Pano_load_image.json ## image-to-pano pipeline, load a square image from local.
    |-- Image2Pano_gen_image.json ## text-to-pano pipeline, generate a square image from a certain SD model. (only support batchsize=1)
    
```
   
