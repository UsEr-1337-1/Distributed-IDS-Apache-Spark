
import numpy as np

if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'string_'):
    np.string_ = bytes
if not hasattr(np, 'unicode_'):
    np.unicode_ = str

# ───────────────────────────────────────────────────────
# 1. Import Spark libraries
# ───────────────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ───────────────────────────────────────────────────────
# 2. Start Spark Session
# ───────────────────────────────────────────────────────
spark = (
    SparkSession.builder
        .appName("DistributedIDS")
        .getOrCreate()
)

print("Spark UI:", spark.sparkContext.uiWebUrl)  # Keep this for screenshot later

# ───────────────────────────────────────────────────────
# 3. Load data
# ───────────────────────────────────────────────────────

train_path = r"C:\Users\username\data\KDDTrain+.txt"
test_path  = r"C:\Users\username\data\KDDTest+.txt"



cols = [*map(str, range(41))] + ["41", "label"]

df = (spark.read
         .csv(train_path, header=False, inferSchema=True)
         .toDF(*cols))

df_test = (spark.read
              .csv(test_path, header=False, inferSchema=True)
              .toDF(*cols))

# 🛠 Drop the "41" column (string type, not needed)
df = df.drop("41")
df_test = df_test.drop("41")

# ───────────────────────────────────────────────────────
# 4. Feature Engineering Pipeline
# ───────────────────────────────────────────────────────
cat_cols = ["1", "2", "3"]  # protocol_type, service, flag (categorical)
num_cols = [c for c in df.columns if c not in cat_cols + ["label"]]

# Index and encode categorical columns
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx") for c in cat_cols]

encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in cat_cols],
    outputCols=[f"{c}_vec" for c in cat_cols]
)

# Assemble all features into one vector
assembler = VectorAssembler(
    inputCols=num_cols + [f"{c}_vec" for c in cat_cols],
    outputCol="features"
)

# Define RandomForest Classifier
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=50,
    seed=42
)

# Create full pipeline
pipeline = Pipeline(stages=[*indexers, encoder, assembler, rf])

# ───────────────────────────────────────────────────────
# 5. Train the model
# ───────────────────────────────────────────────────────
model = pipeline.fit(df)  

# ───────────────────────────────────────────────────────
# 6. Evaluate the model
# ───────────────────────────────────────────────────────
predictions = model.transform(df_test)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

f1_score = evaluator.evaluate(predictions)

print(f"F1-score on test set: {round(f1_score, 4)}")

# ───────────────────────────────────────────────────────
# 7. Save the trained pipeline model
# ───────────────────────────────────────────────────────
#model.save("spark_ids_model")

print("Model saved successfully!")

# ───────────────────────────────────────────────────────
# 8. Stop Spark session
# ───────────────────────────────────────────────────────
spark.stop()
