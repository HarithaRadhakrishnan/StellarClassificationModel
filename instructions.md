This is a project that aims to sort the celestial objects into either a star, a galaxy or a quasar based on their photometric filters. 
This project leverages deep learning techniques to classify celestial objects using photometric data from u, g, r, i, and z spectral bands. Built using TensorFlow and Keras in Google Colab, the model preprocesses raw data by cleaning, normalizing, and encoding it before training a multi-layer neural network for multi-class classification. The final model achieves accurate predictions and is saved as HR.h5 for reuse. Comprehensive evaluation metrics, including accuracy reports and visualization graphs, are provided to analyze the model's performance, demonstrating its utility in astronomical data analysis and classification tasks.
The dataset was taken from the SDSS server.
The DR15 dataset was used for testing and the DR17 dataset was used for validation.
97% accuracy was achieved.
