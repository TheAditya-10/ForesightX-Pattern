Getting started
===============

ForesightX Pattern has one Python package, ``foresightx_pattern``.
The ``ml`` package owns data ingestion, preprocessing, feature engineering,
training, evaluation, and MLflow logging. The ``app`` package owns the FastAPI
inference service.

Install the full training and development environment:

.. code-block:: bash

   pip install -r requirements.txt

Run the DVC training stage:

.. code-block:: bash

   dvc repro

Run the same training entry point directly:

.. code-block:: bash

   python3 -m foresightx_pattern.ml.training.train

Start the API after a model bundle exists in ``artifacts/model``:

.. code-block:: bash

   uvicorn foresightx_pattern.app.main:app --reload --host 0.0.0.0 --port 8003
