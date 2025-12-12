Go to google cloud console,
Create a cloud storage with name "my-stock-project-bucket"
Create a data folder and upload 3 files inside the data folder
Upload the python file in the root of the bucket.

At the top click the terminal icon and let the terminal open

For first time:
gcloud dataproc clusters create stock-cluster \
    --region us-central1 \
    --zone us-central1-a \
    --master-machine-type n1-standard-2 \
    --worker-machine-type n1-standard-2 \
    --num-workers 2 \
    --image-version 2.1-debian11

IF cluster already exists:

gcloud dataproc clusters start stock-cluster --region us-central1



Once the cluster is running:
gcloud dataproc jobs submit pyspark gs://my-stock-project-bucket/stock_prediction_spark.py \
    --cluster=stock-cluster \
    --region=us-central1 \
    -- \
    --bucket=my-stock-project-bucket
