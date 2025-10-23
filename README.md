TP – CNN “from scratch” vs Transfert Learning (Cats vs Dogs)

Objectif
Comparer un modèle CNN entraîné from scratch et un modèle en transfert d’apprentissage sur le même jeu de données (cats vs dogs). Montrer l’impact du transfert learning sur la convergence, la performance et la robustesse.

Environnement
python -m venv .venv
 .venv\Scripts\activate
pip install -r requirements.txt

Voici ce que contient mon requirements.txt:
torch>=2.0
torchvision>=0.15
torchaudio
matplotlib
tqdm
scikit-learn
numpy

Organisation des données ;
CNN-CATSDOGS_MEGANEKASONGO
data/
    test/
        Cat/
        images.jpg
        dog/
        images.jpg
    train/
        Cat/
        images.jpg
        dog/
        images.jpg
project/
    checkpoints/
    
    collect_results.py
    eval.py
    models.py
    train.py
    utils.py
    collect_results.py

runs/
    fromscratch_adam_Ir0.0/
        accuracy.png
        confusion_test.png
        history.csv
        loss.png
        precision_recall.png
    fromscratch_sgd_Ir0.01/
        accuracy.png
        confusion_test.png
        confusion_val.png
        history.csv
        loss.png
        precision_recall.png
    transfer_adam_Ir0.001/
    accuracy.png
        confusion_test.png
        confusion_val.png
        history.csv
        loss.png
        precision_recall.png
.gitignore
requirements.txt
README.md

Voici les commandes pour entraîner l'expérience A:
 1. Entraînement Expérience A (CNN from scratch)

 1. Entraînement Expérience A (CNN from scratch)

%cd /content/project
!python train.py --exp A \
  --data_root /content/data \
  --epochs 20 \
  --batch_size 16 \
  --img_size 224 \
  --lr 1e-3 \
  --optimizer adam \
  --scheduler step \
  --augment \
  --subset 1.0


#  2. Entraînement Expérience B (Transfer Learning ResNet18)

!python train.py --exp B \
  --data_root /content/data \
  --epochs 20 \
  --batch_size 16 \
  --img_size 224 \
  --lr 1e-3 \
  --optimizer adam \
  --scheduler step \
  --augment \
  --pretrained \
  --subset 1.0

résultat 

runs/<run_name>/: loss.png, accuracy.png, precision_recall.png, confusion_val.png (+ confusion_test.png si data/test existe).

Checkpoint (meilleur val_acc) : checkpoints/best_<run_name>.pth.

<run_name> :

fromscratch_<optimizer>_lr<LR> (A)

transfer_<optimizer>_lr<LR> (B)

pour afficher le tableau voici la commande ;
python collect_results.py

Comparaison;
Le transfert learning converge en moins d’époques et atteint une meilleure accuracy que le CNN from scratch. Sur un dataset limité, le backbone pré-entraîné fournit des features plus discriminantes, souvent avec +4 à +8 points d’accuracy, et des courbes de loss plus stables.
Le transfert learning converge en moins d’époques et atteint une meilleure accuracy que le CNN from scratch. Sur un dataset limité, le backbone pré-entraîné fournit des features plus discriminantes, souvent avec +4 à +8 points d’accuracy, et des courbes de loss plus stables.

