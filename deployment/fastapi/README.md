# FastAPI Deployment Notes

Recommended free-path candidate: Render Free Web Service.

Why this is prepared:

- Render currently documents free web services for Python apps.
- Render's FastAPI guide uses `pip install -r requirements.txt` and a `uvicorn` start command.
- This project already has a FastAPI entrypoint at `aqi_predictor.api.main:app`.

Important limitations:

- Free Render services can spin down after inactivity.
- Free services have monthly usage limits.
- Do not add a paid plan or payment method unless explicitly approved by the project owner.
- Do not commit Hopsworks secrets.

Prepared blueprint:

```text
deployment/fastapi/render.yaml
```

If using the Render dashboard manually, use:

```text
Build command:
pip install -r requirements.txt

Start command:
uvicorn aqi_predictor.api.main:app --host 0.0.0.0 --port $PORT
```

Environment variables to configure in Render:

```text
PYTHONPATH=src
HOPSWORKS_API_KEY=<set in Render dashboard, never commit>
HOPSWORKS_PROJECT=<set in Render dashboard>
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
HOPSWORKS_CERT_FOLDER=.hopsworks-certs
```

After deployment, test:

```text
https://<your-render-service>.onrender.com/health
https://<your-render-service>.onrender.com/predict
https://<your-render-service>.onrender.com/docs
```

