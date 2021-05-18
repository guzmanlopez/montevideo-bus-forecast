![logo fing udelar](https://eva.fing.edu.uy/pluginfile.php/1/theme_adaptable/logo/1611344323/Banner%20eva-01.png)

# Curso Aprendizaje Automático para Datos en Grafos

**Docente:** Prof. Gonzalo Mateos (Universidad de Rochester, EEUU).

**Docente invitado:** Fernando Gama (Universidad de California Berkeley, EEUU).

**Otros docentes:** Marcelo Fiori y Federico La Rocca.

**Fechas:** 01/02/2021 al 04/02/2021 y 11/02/2021.

**Web:** [Página principal del curso en plataforma Eva](https://eva.fing.edu.uy/course/view.php?id=1484)

---

## Proyecto final del curso

### Predicción del flujo de pasajeros en las paradas de ómnibus del Sistema de Transporte Metropolitano (STM) de Montevideo

**Estudiante:** Guzmán López

La estructura de este repositorio fue clonada de [Data Science Project Template](https://github.com/makcedward/ds_project_template) y sigue los lineamientos descritos en [Manage your Data Science project structure in early stage](https://towardsdatascience.com/manage-your-data-science-project-structure-in-early-stage-95f91d4d0600).

- **src:** Código fuente en Python y R utilizado en múltiples escenarios. During data exploration and model training, we have to transform data for particular purpose. We have to use same code to transfer data during online prediction as well. So it better separates code from notebook such that it serves different purpose.
- **test:** In R&D, data science focus on building model but not make sure everything work well in unexpected scenario. However, it will be a trouble if deploying model to API. Also, test cases guarantee backward compatible issue but it takes time to implement it.
- **model:** Folder for storing binary (json or other format) file for local use.
- **data:** Folder for storing subset data for experiments. It includes both raw data and processed data for temporary use.
- **notebook:** Todos los notebooks incluyendo el Análisis Exploratorio de Datos y fase de modelado.

### Requerimientos

- Python versión 3.9.1
- R versión 4.0.4

#### Configurar ambiente para reproducir este código: 

- Clonar repositorio

```{sh}
git clone https://github/guzmanlopez/montevideo-bus-forecast
cd montevideo-bus-forecast
```

- **Opcional:** instalar [pyenv](https://github.com/pyenv/pyenv#installation) para obtener la versión de Python requerida

```{sh}
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

# Reiniciar terminal
exec $SHELL

# Instalar Python 3.9.1
pyenv install 3.9.1

# Definir la versión de Python en la ruta donde estoy ubicado
pyenv local 3.9.1

# Habilitar la versión de Python instalada
pyenv shell 3.9.1
```

- Instalar [poetry](https://python-poetry.org/) como gestor de librerías y dependencias de Python

```{sh}
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

source $HOME/.poetry/env

# Iniciar poetry
poetry shell
poetry env use 3.9.1
poetry env info
poetry install
```

- Instalar [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) y [PyTorch Geometric Temporal](https://pytorch-geometric-temporal.readthedocs.io/en/latest/index.html)  

**Nota:** instalación de PyTorch sin soporte de GPU, para soporte con GPU modificar `+cpu`CUDA (Ejemplo: `+cu101`)

```{sh}
# Instalar PyTorch
poetry add torch=1.8.0+cpu --platform linux --python "^3.9"

# Instalar PyTorch Geometric
poetry add torch-scatter --platform linux --python "^3.9"
poetry add torch-sparse --platform linux --python "^3.9"
poetry add torch-cluster --platform linux --python "^3.9"
poetry add torch-spline-conv --platform linux --python "^3.9"
poetry add torch-geometric --platform linux --python "^3.9"

# Instalar PyTorch Geometric Temporal
poetry add torch-geometric-temporal --platform linux --python "^3.9"

# Instalar PyTorch Lightning
poetry add pytorch-lightning
```
