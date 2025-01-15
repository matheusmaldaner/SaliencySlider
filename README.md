# SaliencySlider

SaliencySlider is a web application that enables users to upload images and interactively explore how a pretrained VGG19 convolutional neural network uses specific regions of the image for classification. The application features a saliency slider that allows users to dynamically control the visibility of influential image regions.

## Features

- **Image Upload**: Users can upload their own images to analyze.
- **Interactive Saliency Exploration**: A slider provides control over how many features of the image are shown, based on their influence on the classification decision.
- **Pretrained VGG19 CNN**: Utilizes a robust model trained on a wide variety of images for accurate classification.

## Getting Started

To use SaliencySlider, visit the deployed web application or experiment with the code in Google Colab.

- **Web Application**: [Visit SaliencySlider](https://matheusmaldaner.pythonanywhere.com/GradCam/)
- **Google Colab**: [Experiment with SaliencySlider](https://colab.research.google.com/drive/1xJEuaht0o6cHeA3eo6A3KWSu14sSp-wY?usp=sharing)

## Final Report

For a detailed explanation of the project, methodologies, and results, refer to our final report:

- [Read the Final Report](./SaliencySlider_Report.pdf)

We are currently working on further development and tests for SaliencySlider, exploring the potential direction of conducting a user study to gauge the usability and usefulness of our application.

## Deployment

The SaliencySlider web application is deployed using Django and hosted on pythonanywhere, ensuring consistent availability and performance.

We are currently working on transfering the deployment of the application to be hosted using Docker on Amazon Lightsail Container.

### Docker:
Make sure Docker Daemon is running
Build command:
`docker build -t saliency-slider-app`
Run command: 
`docker run -d -p 8000:8000 saliency-slider-app`


## Contributors

* Matheus Kunzler Maldaner

* Kian Ambrose

* Lexie Certo

* Kristian O'Connor
