# Final Project Deep Learning: 
## Sunthorn Phu Generator
## Introduction

This project is a part of the final project of Deep Learning course at  [<b>Panyapiwat Institute of Management</b>](https://www.pim.ac.th). The project is to generate a poem in the style of Sunthorn Phu, a famous Thai poet, using a deep learning model. The model is trained on a dataset of Sunthorn Phu's poems. The dataset is obtained from [vajirayana.org/](https://vajirayana.org/). The model is trained using a recurrent neural network with LSTM cells. The model is trained to predict the next character given a sequence of characters. The model is trained on Computer :

        Computer: Aspire-A715-42G 
        OS: Kubuntu 22.04.2 LTS x86_64 
        Host: Aspire A715-42G V1.08 
        Kernel: 5.19.0-42-generic 
        Resolution: 1920x1080, 1920x1080 
        CPU: AMD Ryzen 5 5500U with Radeon Graphics (12) @ 2.100GHz 
        GPU: NVIDIA GeForce GTX 1650 Mobile / Max-Q 
        GPU: AMD ATI 05:00.0 Lucienne 
        Memory: 9034MiB / 15327MiB 
        PyTorch version:  2.0.1
        CUDA version:  11.8
        cuDNN version:  8700

## Setup Computer

- Conda create environment pytorchphu
Run the following command to create the Conda environment from the environment.yml file:
``` command
conda env create -f environment.yml
```

- Once the environment is created, activate it using the following command:

```
conda activate pytorchphu
```

## Quick Start
- run code in command
```
    python RunGenerator.py
```
- run on google colab RunGenerator
        [<b>RunGenerator.ipynb</b>](https://colab.research.google.com/drive/1-4CJmxJFg9ortnnj_b83GxIMHK_BbH9g?usp=sharing).
- run on google colab SunthornPytorch
        [<b>SunthornPytorch.ipynb</b>](https://colab.research.google.com/drive/1wJaYyY9gg8wozVv0ElfH3_mVwMjnnOtd?usp=sharing).

## Files
    ├── Dataset
    │   ├── AllData.txt # All Data before normalize (line 65277)
    │   ├── DataNirat
    │   │   ├── Niras.txt
    │   │   ├── นิราศพระบาท.txt
    │   │   ├── นิราศพระประธม.txt
    │   │   ├── นิราศภูเขาทอง.txt
    │   │   ├── นิราศเมืองแกลง.txt
    │   │   ├── นิราศเมืองเพชร.txt
    │   │   ├── นิราศวัดเจ้าฟ้า.txt
    │   │   ├── นิราศอิเหนา.txt
    │   │   └── รำพันพิลาป.txt
    │   ├── DataNitan
    │   │   ├── Cobuut.txt
    │   │   ├── Lugsanawong.txt
    │   │   ├── Phaapaimanee.txt
    │   │   └── Singtaipop.txt
    │   ├── NonSunthorn
    │   │   └── KhunChang.txt 
    │   └── TextAfterNormalize.txt # Success Normalize Text (Line: 65285)
    ├── Dict
    │   ├── char_dict.pkl   # char to int
    │   ├── input_data.pkl # train data
    │   └── int_dict.pkl  # int to char
    ├── environment.yml # file for setup miniconda/anaconda env
    ├── Model
    │   ├── model_info.pth # parameter file
    │   ├── model.pth # model state dict file
    │   └── model_run.pth # model for run (torch.save(model,'...'))
    ├── NormalizeDataset.ipynb # code for normalize text file
    ├── readme.md
    ├── requirements.txt # pip 
    ├── RunGenerator.ipynb # file for run generator
    ├── SunthornPytorch.ipynb # File for create dataset
        , train model, and save model
    └── RunGenerator.py # for run quick start generator and save output.txt


## Member Group 4
- <h5>นายณัชพล สิทธิอาษา 6352300090 </h5>
- <h5>นางสาวจรรยพรพรหม ธีระนันท์ 6352300138 </h5>
- <h5>นายธวัชชัย บัวจันทร์ 6352300197 </h5>
- <h5>นางสาวณัฐนันท์ บุญหมั่น 6352300219 </h5>
- <h5>นางสาวณัฏฐธิดา รายขุนทด 6352300367 </h5>
- <h5>Mr.Sattrawut Piasui 6352300405</h5>


## Reference
### code 
- https://github.com/edumunozsala/Character-Level-Text-Generation/tree/master
### dataset
- โคบุตร [https://vajirayana.org/%E0%B8%99%E0%B8%B4%E0%B8%97%E0%B8%B2%E0%B8%99%E0%B8%84%E0%B8%B3%E0%B8%81%E0%B8%A5%E0%B8%AD%E0%B8%99%E0%B8%AA%E0%B8%B8%E0%B8%99%E0%B8%97%E0%B8%A3%E0%B8%A0%E0%B8%B9%E0%B9%88%E0%B9%80%E0%B8%A3%E0%B8%B7%E0%B9%88%E0%B8%AD%E0%B8%87%E0%B9%82%E0%B8%84%E0%B8%9A%E0%B8%B8%E0%B8%95%E0%B8%A3/%E0%B8%AD%E0%B8%98%E0%B8%B4%E0%B8%9A%E0%B8%B2%E0%B8%A2%E0%B9%80%E0%B8%A3%E0%B8%B7%E0%B9%88%E0%B8%AD%E0%B8%87%E0%B9%82%E0%B8%84%E0%B8%9A%E0%B8%B8%E0%B8%95%E0%B8%A3](https://vajirayana.org/โคบุตร)
- พระอภัยมณี https://vajirayana.org/%E0%B8%9E%E0%B8%A3%E0%B8%B0%E0%B8%AD%E0%B8%A0%E0%B8%B1%E0%B8%A2%E0%B8%A1%E0%B8%93%E0%B8%B5
- ลักษณวงศ์ https://vajirayana.org/%E0%B8%A5%E0%B8%B1%E0%B8%81%E0%B8%A9%E0%B8%93%E0%B8%A7%E0%B8%87%E0%B8%A8%E0%B9%8C
- สิงหไกรภพ https://vajirayana.org/%E0%B8%AA%E0%B8%B4%E0%B8%87%E0%B8%AB%E0%B9%84%E0%B8%81%E0%B8%A3%E0%B8%A0%E0%B8%9E
- นิราศเมืองแกลง https://th.wikisource.org/wiki/%E0%B8%99%E0%B8%B4%E0%B8%A3%E0%B8%B2%E0%B8%A8%E0%B9%80%E0%B8%A1%E0%B8%B7%E0%B8%AD%E0%B8%87%E0%B9%81%E0%B8%81%E0%B8%A5%E0%B8%87
- นิราศพระบาท https://th.wikisource.org/wiki/%E0%B8%99%E0%B8%B4%E0%B8%A3%E0%B8%B2%E0%B8%A8%E0%B8%9E%E0%B8%A3%E0%B8%B0%E0%B8%9A%E0%B8%B2%E0%B8%97
- นิราศภูเขาทอง https://th.wikisource.org/wiki/%E0%B8%99%E0%B8%B4%E0%B8%A3%E0%B8%B2%E0%B8%A8%E0%B8%A0%E0%B8%B9%E0%B9%80%E0%B8%82%E0%B8%B2%E0%B8%97%E0%B8%AD%E0%B8%87
- นิราศวัดเจ้าฟ้า https://th.wikisource.org/wiki/%E0%B8%99%E0%B8%B4%E0%B8%A3%E0%B8%B2%E0%B8%A8%E0%B8%A7%E0%B8%B1%E0%B8%94%E0%B9%80%E0%B8%88%E0%B9%89%E0%B8%B2%E0%B8%9F%E0%B9%89%E0%B8%B2
- นิราศอิเหนา https://th.wikisource.org/wiki/%E0%B8%99%E0%B8%B4%E0%B8%A3%E0%B8%B2%E0%B8%A8%E0%B8%AD%E0%B8%B4%E0%B9%80%E0%B8%AB%E0%B8%99%E0%B8%B2
- รำพันพิลาป https://th.wikisource.org/wiki/%E0%B8%A3%E0%B8%B3%E0%B8%9E%E0%B8%B1%E0%B8%99%E0%B8%9E%E0%B8%B4%E0%B8%A5%E0%B8%B2%E0%B8%9B
