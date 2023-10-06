# Noise Without Earthquakes Detection

This repository contains Python scripts for detecting and analyzing noise in seismic data without earthquake events. It uses a Convolutional Neural Network (CNN) to process seismic data and extract relevant features for noise analysis. The repository is organized as follows:

## Description

This project aims to detect and analyze noise in seismic data, excluding earthquake events. It utilizes a CNN-based model to process seismic data, making it easier to identify and understand noise patterns.

## Getting Started

### Prerequisites

Make sure you have the following prerequisites installed:

- Python 3.x
- Required Python libraries (e.g., NumPy, pandas, TensorFlow, ObsPy)
- ObsPy's FDSN client (for seismic data retrieval)
- Matplotlib (for data visualization)

### Usage

To use the scripts, follow these steps:

1. Clone this repository to your local machine.
2. Install the required Python libraries listed above.
3. Run the scripts with the necessary command-line arguments.

## Parameters

The scripts accept several command-line arguments for configuring the data processing and analysis. Here's a brief description of the parameters:

- `yr`, `mth`, `day`: Year, month, and day of data processing.
- `num`: Number of days of processing.
- `sta`: Network station for processing.
- `cha`: Channel of seismic data.
- `h1`, `h2`: Start and end hours for processing.
- `f1`, `f2`: First and last frequency for the analysis.
- `yr2`, `mth2`, `day2`: Year, month, and day for comparison.
- `num2`: Number of days for comparison.

## Initialization

The initialization section of the script sets up parameters and loads data specific to the chosen network station. It also initializes the AI model for processing.

## Data Processing

This section handles data retrieval, preprocessing, and feature extraction. The AI model processes the data to identify noise patterns.

## AI Model

The AI model used for noise detection is a Convolutional Neural Network (CNN). The model is loaded and used for predictions on the seismic data.

## Plotting PPSD

The script generates Power Spectral Density (PPSD) plots to visualize noise levels across different frequencies. It also calculates and plots the 5th and 95th percentiles.

## Comparing Data

The final section compares the generated PPSD plots with data from another specified date to identify any significant changes in noise levels.

Feel free to customize and adapt these scripts to your specific seismic data analysis needs.

**Note**: Ensure that you have access to the necessary seismic data and appropriate permissions before using these scripts.

For any questions or issues, please [open an issue](https://github.com/your-username/your-repo/issues) on this repository.
