call deactivate

call cd D:\workspace\repos\customer-churn

call .venv\Scripts\activate

call cd deployment


call set MODEL_NAME=logisticreg_2025-06-12_14-06.joblib
call set MODEL_PATH=..\models\logisticreg_2025-06-12_14-06.joblib

call python app/app.py

call deactivate