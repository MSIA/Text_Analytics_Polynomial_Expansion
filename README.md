## The Polynomial Expansion Project

The objective of this project is to implement a deep learning model that learns to expand single variable polynomials, where the model takes the factorized sequence as input and predict the expanded sequence without parsing or using rule-based methods.

The transformer emerged as the best performing model.This ML model is exposed as a REST service which accepts as input the factorized polynomial as text and outputs a json containing the ML result i.e. the expanded polynomial.

#### Steps to run the REST service

1. Clone this repository.
  ```bash
  git clone https://github.com/MSIA/sgk1336_Text_Analytics_Polynomial_Expansion.git
  ```
2. Open the Transformer directory and install dependencies using this command.
  ```bash
  cd Tansformer
  pip install -r requirements.txt --user
  ```
3. To run the REST API service, run the following command.
  ```bash
  python3 api.py
  ```
4. The web app must be running on the local host. To interact, open this link: http://127.0.0.1:5000/apidocs. 


