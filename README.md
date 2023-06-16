# Matrix Capsules with EM Routing
A PyTorch implementation of [Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb)

## Usage
1. Instalar [PyTorch](http://pytorch.org/)

2. Start training 
```bash
python train.py --batch-size 20 --test-batch-size 20
```


## Experimentación con retinografías a color

Hiperparametros especificos `lr=0.01`, `batch_size=20`, `weight_decay=0`, Adam optimizer, sin data augmentation.

El tamaño de los kernels de las capas convolucionales es de 2x2 en lugar de 3x3

Obtenemos los siguientes resultados

| Arch | EM-Iters | Coord Add | Loss | Epochs | Test Accuracy |
| ---- |:-----:|:---------:|:----:|:--:|:-------------:|
| A=32 B=4 C=4 D=4        | 2 | Y | Spread    | 300 |  97.0000  |




## Referencias
- https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow
- https://github.com/gyang274/capsulesEM
- https://github.com/yl-1993/Matrix-Capsules-EM-PyTorch
- https://github.com/shzygmyx/Matrix-Capsules-pytorch
- https://github.com/ahmedshoaib/caps-em-routing-cifar
# PyTorch_RetinografiasColor
