# Auto Tagging for Fashion Retail Using Multi-label Image Classification
---
[![Source Code](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/133hKQ2tNsBmBy3luvEB3IwzVMV66B0kx?usp=sharing)
[![Docker Image](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/repository/docker/danielsyahputra13/pytorch_image_tagging)

## How to Run This App

### Via Cloning This Repo

- Clone this repo: `https://github.com/danielsyahputra/pytorch-image-tagging.git`
- Change the working directory: `cd pytorch-image-tagging`
- Install dependencies: `pip install -r requirements.txt`
- Run thi command: `streamlit run App.py`

### Via Docker

The image is avaliable at [Docker hub](https://hub.docker.com/repository/docker/danielsyahputra13/pytorch_image_tagging). To run this app, you can do the following commands.

- Pull image: `docker pull danielsyahputra13/pytorch_image_tagging`
- Run docker container locally: `docker run -it -rm -p {LOCAL_PORT}:8501 danielsyahputra13/streamlit_word_segmentation`
- Open the browser and go to `localhost:{LOCAL_PORT}` to see the application.

Once all the process has successfully done, you will see the app look like this.

<img src="https://i.ibb.co/djbVtb2/Screen-Shot-2022-09-14-at-21-52-26.png">

## Citation

```
@inproceedings{lefakis2018feidegger,
  title={FEIDEGGER: A Multi-modal Corpus of Fashion Images and Descriptions in German},
  author={Lefakis, Leonidas and Akbik, Alan and Vollgraf, Roland},
  booktitle = {{LREC} 2018, 11th Language Resources and Evaluation Conference},
  year      = {2018}
}
```
