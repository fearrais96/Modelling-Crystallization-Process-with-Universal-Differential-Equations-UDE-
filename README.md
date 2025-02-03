# Hybrid Model for Potassium Sulfate Batch Crystallization
> The aim of this project was develop a hybrid model for the potassium sulfate batch crystallization. The Universal Differential Equations (UDE) hybrid approach was used to develop this model. The nucleation, growth and dissolution rates were replaced by neural networks and combined to the population balance equations to predict the state variables.

## 📖 Overview
This repository contains the source code for State Hybrid Model for Potassium Sulfate Batch Crystallization, developed by Fernando Arrais Romero Dias Lima, Carine M. Rebello, Erbet A. Costa,
Vinícius V. Santana, Marcellus G.F. de Moares, Amaro G. Barreto Jr., Argimiro R. Secchi, Maurício B. de Souza Jr. and Idelfonso B.R. Nogueira at the Laboratory of Software Development for Process Control and Optimization (LADES) - COPPE/UFRJ, in association with the Norwegian University of Science and Technology (NTNU).

The purpose of this project is to develop a hybrid model code in Python and apply it to model nucleation, crystal growth and dissolution for the potassium sulfate batch crystallization.

## 🚀 Features
- Model replacing the growth (G) and nucleation (B) rates by neural networks
- Model replacing the growth (G) rate by neural network
- Model replacing the nucleation (B) rate by neural network
- Model replacing the dissolution rate (D) by neural network

## 📦 Installation
To install and use this project, follow these steps:

### Prerequisites
Ensure you have the following dependencies installed:
```bash
# Example for Python projects
pip install -U scikit-learn
pip install numpy
pip install pandas
```

### Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/LADES-PEQ/Modelling-Crystallization-Process-with-Universal-Differential-Equations-UDE-.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Modelling-Crystallization-Process-with-Universal-Differential-Equations-UDE-
   ```
3. Run the main script:
   ```bash
   UDE_B_G_train.py
   ```
4. (Optional) Configure environment variables or additional settings.

## 📂 Repository Structure
```
├── Data                        # Folder with the data to develop the models
├── DE_B_G_train.py             # Model replacing the growth (G) and nucleation (B) rates by neural networks
├── DE_B_train.py               # Model replacing the nucleation (B) rate by neural network
├── DE_G_train.py               # Model replacing the growth (G) rate by neural network
├── DE_diss_train.py            # Model replacing the dissolution rate (D) by neural network
├── LICENSE                     # License information
└── README.md                   # Project documentation
```

## ✏️ Authors & Contributors
This project was developed by **Laboratory of Software Development for Process Control and Optimization (LADES) - COPPE/UFRJ** under the coordination of **Argimiro Resende Secchi** in collaboration with the **Process System Engineering (PSE) group** from the **Norwegian University of Science and Technology (NTNU)**.

- **Fernando Arrais Romero Dias Lima** - Code development and paper writing - farrais@eq.ufrj.br
- **Carine M. Rebello** - Code development and paper writing
- **Erbet A. Costa** - Code development and paper writing
- **Vinícius V. Santana** - Paper writing
- **Marcellus G.F. de Moares** - Paper writing
- **Amaro G. Barreto Jr.** - Supervision
- **Argimiro Resende Secchi** - Supervision
- **Maurício Bezerra de Souza Jr.** - Supervision
- **Idelfonso B.R. Nogueira** - Supervision

We welcome contributions!

## 🔬 References & Publications
If you use this work in your research, please cite the following publications:
- **Fernando Arrais Romero Dias Lima, Carine M. Rebello, Erbet A. Costa, Vinícius V. Santana, Marcellus G.F. de Moares, Amaro G. Barreto Jr., Argimiro R. Secchi, Maurício B. de Souza Jr. and Idelfonso B.R. Nogueira. "Improved modeling of crystallization processes by Universal Differential Equations." Chemical Engineering Research and Design 2023, 200, 538-549, https://doi.org/10.1016/j.cherd.2023.11.032 .**
- **GitHub Repository**: https://github.com/LADES-PEQ/Modelling-Crystallization-Process-with-Universal-Differential-Equations-UDE-.git

BibTeX:
```bibtex
@article{lima2023cherd,
title = {Improved modeling of crystallization processes by Universal Differential Equations},
journal = {Chemical Engineering Research and Design},
volume = {200},
pages = {538-549},
year = {2023},
issn = {0263-8762},
doi = {https://doi.org/10.1016/j.cherd.2023.11.032},
author = {Fernando Arrais R.D. Lima and Carine M. Rebello and Erbet A. Costa and Vinícius V. Santana and Marcellus G.F. de Moares and Amaro G. Barreto and Argimiro R. Secchi and Maurício B. de Souza and Idelfonso B.R. Nogueira}
}
```

## 🛡 License
This work is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0) License**.  
You are free to:
- **Use, modify, and distribute** this code for any purpose.
- **Cite the following reference** when using this code:

  **Fernando Arrais Romero Dias Lima, Carine M. Rebello, Erbet A. Costa, Vinícius V. Santana, Marcellus G.F. de Moares, Amaro G. Barreto Jr., Argimiro R. Secchi, Maurício B. de Souza Jr. and Idelfonso B.R. Nogueira**.  
  **"Improved modeling of crystallization processes by Universal Differential Equations"**,  
  *Chemical Engineering Research and Design*, vol. 200, p. 538-549, 2023.  
  [DOI: https://doi.org/10.1016/j.cherd.2023.11.032]

See the full license details in the LICENSE.txt.

## 📞 Contact
For any inquiries, please contact **Fernando Arrais Romero Dias Lima** at **farrais@eq.ufrj.br** or open an issue in this repository.

## ⭐ Acknowledgments
We acknowledge the support of **Federal University of Rio de Janeiro**, **Capes**, and all contributors.
