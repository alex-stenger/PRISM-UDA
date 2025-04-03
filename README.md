# PRISM-UDA

## Environment Setup

First, please install cuda version 11.0.3 available at [https://developer.nvidia.com/cuda-11-0-3-download-archive](https://developer.nvidia.com/cuda-11-0-3-download-archive). It is required to build mmcv-full later.

For this project, we used python 3.8.5. We recommend setting up a new virtual
environment:

```shell
python -m venv ~/venv/prism-uda
source ~/venv/prism-uda/bin/activate
```

In that environment, the requirements can be installed with:

```shell
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

Please, download the MiT-B5 ImageNet weights provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training)
from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `pretrained/`.

## Adding a new experiment 

To create a new experiment (for instance, on a new datasets), you will have to create two new file :
- a config file, following the example that you can find in ```configs/mic/sample_config.py```and by correctly replacing ```NAME_OF_DATASET_FILE.py```, ```EXPERIMENT_NAME```and ```DATASET_NAME```.
- a dataset file, following the example that you can find in ```configs/_base_/datasets/sample_datasets.py``` and by correcly replacing ```ROOT_TO_SOURCE_DATASET``` and ```ROOT_TO_TARGET_DATASET``` 

## Training a new model
First of all, be careful to properly activate the venv with the following command :
```source ~/venv/prism-uda/bin/activate```

Once you have created the correct config and dataset file, you can run a new experiment by using 
```python run_experiments.py --config configs/mic/config_file.py```
The training scripts create a working directory in ```work_dirs/local-basic```

## Infering after training
In order to predict on the target domain after training, please use the following shell script :
```shell
sh test.sh work_dirs/local-basic/run_name/
```

## Getting the results after inference 
```
python get_results.py --pred_path work_dirs/local-basic/run_name/preds/ --gt_path path-to-labels-ground-truth
```

# Instructions pour Mouhamed

## Relancer les expériences 
Commence par relancer les experiences WeiH->FS1 et FS2->FS1 pour prendre en main l'entrainement, l'inférence, et le code qui sort les résultats.
ATTENTION : dans les données que je t'ai transféré, FS1 s'appelle i3 et FS2 s'apelle lw4
ATTENTION2 : Pour ce cas là, les fichiers de configuration et de datasets (dans le dossier configs) existent déjà, à toi de trouver les bons

## Faire de nouvelles expériences sur GTA5-Cityscapes 

Pour télécharger :

**Cityscapes:** Please, download leftImg8bit_trainvaltest.zip and
gt_trainvaltest.zip from [here](https://www.cityscapes-dataset.com/downloads/)
and extract them to `data/cityscapes`.

**GTA:** Please, download all image and label packages from
[here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract
them to `data/gta`.

**Première étape** : Lancer SAM sur ces deux datasets, et mettre les résultats dans un sous-dossier "sam" (à la manière des autres données que je te donne, il suffit de regarder la structure des fichiers et dossiers pour comprendre

**Deuxième étape** : Adapter les fichiers de pre-processing disponibles dans ```tools/convert_datasets```.
Il faut adapter les fichiers ```tools/convert_datasets/cityscapes.py``` et ```tools/convert_datasets/gta.py``` en s'inspirant des fichier ```tools/convert_datasets/I3-LW4.py``` et ```tools/convert_datasets/I3-LW4_multi.py```.
Ce sont des codes de pré-processing des données essentiels au bon fonctionnement du code d'entraînement.
Une fois que ces codes seront adaptés, il faudra les lancer sur les datasets cityscapes et gta5 avec les commandes suivantes :
```
python tools/convert_datasets/gta_modified.py data/gta --nproc 8
python tools/convert_datasets/cityscapes_modified.py data/cityscapes --nproc 8
```

**Troisième étape** : Adapter l'entraînement, nottament à regarder précisément le fichier dans lequel tout se passe :
```mmseg/models/uda/dacs.py```
C'est dans ce fichier que tu trouveras tous les modules bout à bout, et nottament le réseau de raffinage, qui sera à modifier niveau nombre de classes et autres.

**Étape finale** :
Une fois tout cela fait, tu lances l'entrainement, et tu récupère les résultats.


## Objectif long terme 
Améliorer la méthode actuelle, n'hésite pas à faire des propositions si tu vois des choses à améliorer.



## Acknowledgements

PRISM-UDA is based on the following open-source projects. We thank their
authors for making the source code publicly available.

* [MIC](https://github.com/lhoyer/MIC)
* [HRDA](https://github.com/lhoyer/HRDA)
* [DAFormer](https://github.com/lhoyer/DAFormer)
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [SegFormer](https://github.com/NVlabs/SegFormer)
* [DACS](https://github.com/vikolss/DACS)
