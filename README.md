# Machine Learning for Data Researchers 🔬🧪

Welcome! This repository contains introductory materials, mathematical frameworks, and practical data processing modules designed for laboratory scientists, physicist, chemists, and engineering students.

## 📋 Table of Contents
* **[Interactive Presentation Slides](https://llinersy.github.io/machine-learning-tutorial-for-experiments/module1_Intro/ml_for_experimentalists.html)** - A conceptual framework translating everyday life examples into mathematical explanations.
* **Module 1:** Intro to ML for Experiments, interactive slides.
* **Module 2:** Data-Processing for Spectroscopy & Signal Filtering
* **Module 3:** Decoupling High-Dimensional Data Matrices via PCA (Principal Component Analysis) and t-SNE (t-distributed Stochastic Neighbor Embedding)
* **Module 4:** Image Processing & SEM Grain Morphometry (SEM/TEM of COFs, grain images, porosity, etc).
* **Module 5:** 1D-CNN Signals & 2D-CNN for Spectra Analysis and Micrograph Reticulation Analysis.
* **Module 6:** Integration Projects. CNN for polymerization analysis. PCA for Environmental Analytics & Microplastics Research. Filters for advanced materials characterization.

## 🚀 Student Guide: Running Notebooks in Google Colab with Google Drive

Follow these step-by-step instructions to configure your cloud environment, load the course modules, and link your file directories.

---

### Phase 1: Cloud Folder Setup

1. Open your **Google Drive** ([drive.google.com](https://drive.google.com/)).
2. At the absolute **root** of your Drive (inside `My Drive`), create a new folder named exactly:
   ```text
   IA-Tutorial-2026  
*⚠️ Note: Do not change the casing or spacing. The computer is case-sensitive!*

3. Download the target Python Notebook file (e.g., module2.ipynb) from this GitHub repository to your local computer.
4. Drag and drop that downloaded .ipynb notebook file directly into your newly created IA-Tutorial-2026 Google Drive folder.

### Phase 2: Launching the Notebook
5. Inside your Google Drive folder, right-click on the notebook file (module2.ipynb).
6. Hover over Open with and select Google Colaboratory.

*💡 Troubleshooting Tip for Students: If you do not see Google Colaboratory in the list, click Connect more apps, search for "Colaboratory", click install/connect, and try right-clicking the file again.*

### Phase 3: Mounting Google Drive & Navigating Path Directories
Once your notebook initializes in the browser, you must grant it access to read and write data directly from your Google Drive folder.

7. Look at the left vertical sidebar panel in Colab and click on the Files (Folder icon) tab.
8. Click the Mount Drive icon (a folder icon overlaid with a Google Drive triangle logo).
   - Alternatively, you can create a fresh code cell at the absolute top of your notebook and execute this script command:
     ```from google.colab import drive```
     ```drive.mount('/content/drive')```
9. Follow the pop-up permissions prompt window: select your Google account, scroll down, and click Allow.

### Phase 4: Setting the Active Working Environment
To ensure the notebook can locate your data sets, images, and modules automatically without crashing, you must point Colab directly into your project directory folder.

10. Create a new code cell, paste the following system line, and execute it:
    ```%cd /content/drive/MyDrive/IA-Tutorial-2026```
    
*🚨 CRITICAL RULE: Do not forget the percentage sign (%)! This is a special Jupyter magic command that permanently changes the directory pathway. Using a exclamation mark (!cd) will change the path only temporarily for that single line and fail.*

11. To double-check and verify that your working directory pathway successfully shifted to your folder, create another code cell and type:
    
    ```%pwd```
    
    *(Print Working Directory). The output displayed directly below the cell should read exactly:*
    
    ```/content/drive/MyDrive/IA-Tutorial-2026```


📥 **[Download Module 1: Introduction to Machine Learning (PDF)](https://github.com/llinersy/machine-learning-tutorial-for-experiments/raw/main/module1_Intro/Intro_to_Machine_Learning.pdf)**

📥 **[Download Module 2: Data Processing (PDF)](https://github.com/llinersy/machine-learning-tutorial-for-experiments/raw/main/module2_Data-Processing/module2_explanation/module2_explanation.pdf)**

📥 **[Download Module 3: PCA (PDF)](https://github.com/llinersy/machine-learning-tutorial-for-experiments/raw/main/module3_PCA/module3_explanation/module3_explanation.pdf)**

📥 **[Download Module 4: Morphometry I (PDF)](https://github.com/llinersy/machine-learning-tutorial-for-experiments/raw/main/module4_Morphometry/module4-COFs/module4-COFs_explanation/module4-COFs_explanation.pdf)**

📥 **[Download Module 4: Morphometry II (PDF)](https://github.com/llinersy/machine-learning-tutorial-for-experiments/raw/main/module4_Morphometry/module4-SEM-grains/module4-SEM-grains-explanation/module4-SEM-grains-eplanation.pdf)**

📥 **[Download Module 5: CNN I (PDF)](https://github.com/llinersy/machine-learning-tutorial-for-experiments/raw/main/module5_CNN/module5-1_CNN-spectra/module5-1_explanation/module5-1_explanation.pdf)**

📥 **[Download Module 6: CNN II (PDF)](https://github.com/llinersy/machine-learning-tutorial-for-experiments/raw/main/module5_CNN/module5-CNN-polymerization/module5-CNN-images_explanation/module5-CNN-images_explanation.pdf)**

📥 **[Download Module 6: Environmental Analytics & Microplastics (PDF)](https://github.com/llinersy/machine-learning-tutorial-for-experiments/raw/main/module6_Integration/module6-1/Module6-1_explanation/Module6-2_explanation.pdf)**

📥 **[Download Module 6: Advanced Micrograph Analysis (PDF)](https://github.com/llinersy/machine-learning-tutorial-for-experiments/raw/main/module6_Integration/module6-2/module4-SEM-grains-advanced-explanation/module4-SEM-grains-advanced_explanation.pdf)**

