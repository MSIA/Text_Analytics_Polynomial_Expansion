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

#### Steps to run inference

1. To run inference on user input, you can open the `/Factorized_Input` method and select the button that says "Try it out" on the top right corner. 
2. Then you can input the factorized polynomial in the text box inside this method. Then select "Execute" on the bottom of the method to run the prediction. The response will show the expanded polynomial as output. 

Example inputs - 
1. (4-2\*x)\*(5\*x-7)
2. (-9\*sin(n)-31)\*(8\*sin(n)-9)
3. z*(2\*z+18)
