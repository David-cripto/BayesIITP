# BayesIITP
Schrodinger Bridge realization for toy datasets

# Quick start 🚀

```bash
git clone https://github.com/David-cripto/BayesIITP.git
cd BayesIITP
git submodule update --init --recursive
pip install .
```


# Pretrain models

The arrow indicates the forward process of transferring one data distribution $p_{\mathcal{A}}$ to another $p_{\mathcal{B}}$ in terms of the [I2SB](https://arxiv.org/abs/2302.05872).

|     Datasets     | Model |
|:----------------:|:-----:|
| S_curve -> Swiss |  [link](https://drive.google.com/drive/folders/1obJa-SKdpDfR8DICaB3LekpL0-k-bXfO?usp=sharing) |

Example of the trained model:
![image](/assets/trained_SB.gif)

# Train script
To easily start model training, set(more details in the script):

```bash
python train2d.py --name test --dataset1 swiss --dataset2 scurve --ckpt_path ./ --path_to_save ./ --microbatch 256
```