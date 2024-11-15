# CFDS Homework 7

This repository contains the 6th homework for the "Computing for Data Science" class.

**Authors:**  
Simon Vellin & Aleksandr Smolin, BSE DSDM 2024-2025 students

---

## Application Launch

To run the FastAPI application, use the following command:

```bash
uvicorn api_app_main:app --reload
```

---

## API Commands

### Train Pre-Configured Model
To train the pre-configured model, execute the following command:

```bash
curl -X POST http://127.0.0.1:8000/train_model
```

### Predict using .py File
To make predictions using .py file (which should include data to predict):

```bash
python3 run_request.py
```

### Predict for Data from File
To make predictions using data from the `test_request.json` file:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @test_request.json
```

### Predict Directly for Input Data
To make predictions directly for input data (example):

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '[
    {
      "age": 45,
      "height": 170,
      "weight": 70,
      "aids": 1,
      "cirrhosis": 0,
      "hepatic_failure": 0,
      "immunosuppression": 1,
      "leukemia": 0,
      "lymphoma": 0,
      "solid_tumor_with_metastasis": 1
    }
]'
```

---

## Content Description

- **`class*.py` files**  
  These files contain a library with classes and methods for machine learning workflows.

- **`model.py` file**  
  This file contains the main pipeline for running the application.

- **`api_app_main.py` file**  
  This file contains the necessary methods to run the API locally.

---

**Note:** Ensure the FastAPI application is running and accessible on `http://127.0.0.1:8000` before using the commands.
