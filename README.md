# Dataset

The dataset contains data acquired from HySpex VNIR-1800 (Norsk Elektro Optikk, Norway), with a wavelength from 406.8 to 995.8 nm. The hyperspectral tiff image data stored in the `dataset` folder.


# Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```
# Runnig

Before training the model, you need to preprocess the hyperspectral tiff image data. Run the `prepare_data.py` script to extract the grape berry from the tiff hyperspectral images. The script averages the spectral values of all pixels in the grape regions and saves the processed data as a CSV file.

Next, train the WGAN-GP model using the `train_GAN.py` script. This script will load the preprocessed data and train the model. The trained model pth will be saved in the `model12` folder.

After training the WGAN-GP model, you can use the trained model (`model.pth`) to generate new data. Run the `generate_data.py` script to load the trained model and generate synthetic data.

# Citations

If you use our framework, model, or predictions for any academic work, please cite

```bash
@article{lyu2025synthetic,
  title={Synthetic hyperspectral reflectance data augmentation by generative adversarial network to enhance grape maturity determination},
  author={Lyu, Hongyi and Grafton, Miles and Ramilan, Thiagarajah and Irwin, Matthew and Sandoval, Eduardo},
  journal={Computers and Electronics in Agriculture},
  volume={235},
  pages={110341},
  year={2025},
  publisher={Elsevier}
}
```
