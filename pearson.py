import math
import operator
from sys import argv
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
import pyspark

spark = SparkSession.builder.appName("spark-app").getOrCreate()

# Load in data
reviews = spark.read.csv(f'hdfs://{argv[1]}:9000/user/root/historical_reviews')
reviews = reviews.toDF('id', 'asin', 'overall', 'reviewText')
metadata = spark.read.option("header", "true").json(f"hdfs://{argv[1]}:9000/meta.json")

# Feature extraction
avg_length = reviews.select(f.col('asin'), f.length(f.col('reviewText'))) \
        .groupby('asin') \
        .agg(f.avg('length(reviewText)').alias('avg_review_length'))

# Join on asin and clean no reviews
prices = metadata.select(f.col('asin'), f.col('price'))
avg_length_join_metadata = avg_length.join(prices, on='asin')
lr_data = avg_length_join_metadata.dropna().drop('asin')

# Pearson Correlation computation
rt = lr_data.rdd.map(lambda a: (a["avg_review_length"], a['price']))
pairs_count = rt.map(lambda r: 1).reduce(operator.add)
x_sum = rt.map(lambda r: r[0]).reduce(operator.add)
x_pow = rt.map(lambda r: r[0] * r[0]).reduce(operator.add)
y_sum = rt.map(lambda r : r[1]).reduce(operator.add)
y_pow = rt.map(lambda r: r[1]*r[1]).reduce(operator.add)
xy_sum = rt.map(lambda r: r[0] * r[1]).reduce(operator.add)
x_mean = (x_sum/pairs_count)
y_mean = (y_sum/pairs_count)
above = (pairs_count * xy_sum) -(x_sum*y_sum)
below_left = math.sqrt((pairs_count*x_pow)-(pow(x_sum,2)))
below_right = math.sqrt((pairs_count*y_pow)-(pow(y_sum,2)))
r = above/(below_left*below_right)

# Log Pearson Correlation to console and save the rdd
print("pearson correlation = " + str(r))
# Write out RDD as csv because all the values are floats
rt.toDF().write.format('csv').save(f"hdfs://{argv[1]}:9000/pearson_full.csv")

# Ignore below
#rt.saveAsTextFile (f"hdfs://{argv[1]}:9000/pearson_full")