from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.shell import sc
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StringType, ArrayType, FloatType
from sys import argv
from pyspark.sql.functions import udf, count, struct, lit, monotonically_increasing_id

spark = SparkSession.builder.appName("spark-app").getOrCreate()

'''
Load data from HDFS
'''
reviews = spark.read.csv(f'hdfs://{argv[1]}:9000/user/root/historical_reviews')
reviews = reviews.toDF('id', 'asin', 'overall', 'reviewText')
reviews = reviews.fillna({'reviewText': ''}) # Replace nulls with blank string
reviews = reviews.withColumn("uid", monotonically_increasing_id()) # Create Unique ID
# reviews.printSchema()

# INPUT FROM SQOOP (No column headers)
#  |-- INDEX 0: _c0: string (nullable = true)
#  |-- INDEX 1: asin: string (nullable = true) Unique identifier for a review
#  |-- INDEX: helpful: array (nullable = true) helpfulness rating of the review first num/second num
#       |_ NOTE: helpful is an array that is split into 2 strings
#       |_ INDEX 2: Helpful first element 
#       |_ INDEX 3: Helpful second element
#  |-- INDEX 4: overall: string (nullable = true) Rating ** Important for sentiment analysis
#  |-- INDEX 5: reviewText: string (nullable = true)
#  |-- INDEX 6: reviewTime: string (nullable = true)
#  |-- INDEX 7: reviewerID: string (nullable = true)
#  |-- INDEX 8: reviewerName: string (nullable = true)
#  |-- INDEX 9: summary: string (nullable = true)
#  |-- INDEX 10: unixReviewTime: string (nullable = true)
reviews = reviews.dropna()
reviews.groupBy('overall').agg(count('asin').alias('count')).sort('overall').show() 
# +-------+------+
# |overall| count|
# +-------+------+
# |   null|   196|
# |      1| 23018|
# |      2| 34130|
# |      3| 96194|
# |      4|254013|
# |      5|575264|
# +-------+------+
def cleanup_text_format(record):
    ''' Reference to a specific column and split into strings

    param: Takes in 
    '''
    text = record[3]  # The 3rd column corresponds to the review text
    words = text.split()
    return words

udf_formattext = udf(cleanup_text_format, ArrayType(StringType()))
clean_text = reviews.withColumn("reviewTextArray", udf_formattext(struct([reviews[x] for x in reviews.columns])))

# Count Vectorizor Convert a collection of text documents to vectors of token counts
cv = CountVectorizer(inputCol="reviewTextArray", outputCol="rawFeatures", vocabSize = 1000)
cvmodel = cv.fit(clean_text)
featurizedData = cvmodel.transform(clean_text)

vocab = cvmodel.vocabulary
vocab_broadcast = sc.broadcast(vocab)

def map_termID_to_Word(termIndices):
    ''' Map each term index back to its original word
    param (list of integers): Each element correponds to a word represented by an integer
    returns (list of str): Returns the words which are converted from their respective strings
    '''
    return [vocab_broadcast.value[termID] for termID in termIndices]

# Produce Raw TFIDF results
idf = IDF(inputCol="rawFeatures", outputCol="features_tf_idf")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Cleaning and formatting TFIDF data
vector_udf = udf(lambda vector: vector.indices.tolist(), ArrayType(IntegerType()))
rescaledData=rescaledData.withColumn('feature_indices', vector_udf(rescaledData.features_tf_idf))
slen = udf(lambda s: len(s), IntegerType())
rescaledData = rescaledData.withColumn("feature_indices_size", slen(rescaledData.feature_indices))
vector_udf1 = udf(lambda vector: vector.values.tolist(), ArrayType(FloatType()))
rescaledData=rescaledData.withColumn('feature_values', vector_udf1(rescaledData.features_tf_idf))
vector_udf_size = udf(lambda vector: vector.size, IntegerType())
rescaledData=rescaledData.withColumn('total_features', vector_udf_size(rescaledData.features_tf_idf))
udf_map_termID_to_Word = udf(map_termID_to_Word , ArrayType(StringType()))
rescaledData = rescaledData.withColumn("feature_indices_words", udf_map_termID_to_Word(rescaledData.feature_indices))

# Uncomment to see the full data (Comments below show data if from CSV)
# rescaledData.show()
'''
+---+----------+-------+-------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------+---------------------+
|_c0|      asin|helpful|overall|          reviewText|          reviewTime|          reviewerID|        reviewerName|             summary|      unixReviewTime|uid|     reviewTextArray|         rawFeatures|     features_tf_idf|     feature_indices|feature_indices_size|      feature_values|total_features|feature_indices_words|
+---+----------+-------+-------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+---+--------------------+--------------------+--------------------+--------------------+--------------------+--------------------+--------------+---------------------+
|  0|B000F83SZQ| [0, 0]|      5|I enjoy vintage b...|          05 5, 2014|      A1F6404F1VG29J|          Avidreader|  Nice vintage story|          1399248000|  0|[I, enjoy, vintag...|(1000,[0,1,2,3,4,...|(1000,[0,1,2,3,4,...|[0, 1, 2, 3, 4, 7...|                  34|[0.7702577, 0.529...|          1000| [the, and, a, to,...|
|  1|B000F83SZQ| [2, 2]|      4|This book is a re...|          01 6, 2014|       AN0N05A9LIJEQ|            critters|        Different...|          1388966400|  1|[This, book, is, ...|(1000,[0,1,2,3,4,...|(1000,[0,1,2,3,4,...|[0, 1, 2, 3, 4, 5...|                  38|[0.5776933, 0.529...|          1000| [the, and, a, to,...|
|  2|B000F83SZQ| [2, 2]|      4|This was a fairly...|          04 4, 2014|       A795DMNCJILA6|                 dot|               Oldie|          1396569600|  2|[This, was, a, fa...|(1000,[0,1,2,3,4,...|(1000,[0,1,2,3,4,...|[0, 1, 2, 3, 4, 5...|                  34|[0.5776933, 0.794...|          1000| [the, and, a, to,...|
|  3|B000F83SZQ| [1, 1]|      5|I'd never read an...|         02 19, 2014|      A1FV0SX13TWVXQ|"Elaine H. Turley...|  I really liked it.|          1392768000|  3|[I'd, never, read...|(1000,[0,4,5,9,23...|(1000,[0,4,5,9,23...|[0, 4, 5, 9, 23, ...|                  15|[0.19256443, 0.33...|          1000| [the, I, of, this...|

'''
# Select only the relevant columns
final_outputs = rescaledData.select("id", "reviewText", "feature_indices_size", "feature_indices", "feature_indices_words", "feature_values", "overall")
final_outputs.show(truncate=True)
'''
+--------------------+--------------------+--------------------+---------------------+--------------------+-------+
|          reviewText|feature_indices_size|     feature_indices|feature_indices_words|      feature_values|overall|
+--------------------+--------------------+--------------------+---------------------+--------------------+-------+
|I enjoy vintage b...|                  33|[0, 1, 2, 3, 4, 7...| [the, and, to, a,...|[0.7276272, 0.464...|      5|
|This book is a re...|                  37|[0, 1, 2, 3, 4, 5...| [the, and, to, a,...|[0.5457204, 0.464...|      4|
|This was a fairly...|                  33|[0, 1, 2, 3, 4, 5...| [the, and, to, a,...|[0.5457204, 0.696...|      4|
|I'd never read an...|                  15|[0, 4, 5, 8, 22, ...| [the, I, of, this...|[0.1819068, 0.301...|      5|
|If you like perio...|                  14|[0, 5, 8, 16, 42,...| [the, of, this, y...|[0.1819068, 0.441...|      4|
|A beautiful in-de...|                  19|[0, 3, 5, 6, 12, ...| [the, a, of, is, ...|[0.1819068, 0.570...|      4|
|I enjoyed this on...|                  24|[0, 1, 4, 7, 8, 9...| [the, and, I, in,...|[0.1819068, 0.232...|      4|
|Never heard of Am...|                  31|[0, 1, 2, 3, 4, 5...| [the, and, to, a,...|[0.5457204, 0.232...|      4|
|Darth Maul workin...|                  14|[3, 5, 6, 10, 20,...| [a, of, is, that,...|[0.28525183, 0.44...|      5|
|This is a short s...|                  43|[0, 2, 3, 6, 7, 8...| [the, to, a, is, ...|[0.3638136, 0.819...|      4|
|I think I have th...|                  16|[1, 3, 4, 6, 7, 8...| [and, a, I, is, i...|[0.23224682, 0.28...|      5|
|Title has nothing...|                  33|[0, 2, 4, 7, 11, ...| [the, to, I, in, ...|[0.3638136, 1.092...|      4|
|Well written. Int...|                  18|[2, 3, 6, 29, 48,...| [to, a, is, his, ...|[0.8196134, 0.285...|      3|
|Troy Denning's no...|                 108|[0, 1, 2, 3, 5, 6...| [the, and, to, a,...|[4.00195, 2.32246...|      3|
|I am not for sure...|                  36|[0, 2, 3, 4, 5, 6...| [the, to, a, I, o...|[0.909534, 0.2732...|      5|
|I really enjoyed ...|                  12|[0, 4, 46, 49, 53...| [the, I, really, ...|[0.5457204, 0.301...|      5|
|Great read enjoye...|                  18|[4, 5, 8, 10, 12,...| [I, of, this, tha...|[0.30167994, 0.44...|      5|
|Another well writ...|                  32|[0, 1, 2, 3, 4, 5...| [the, and, to, a,...|[0.1819068, 0.464...|      3|
|This one promises...|                  18|[0, 1, 2, 4, 8, 1...| [the, and, to, I,...|[0.1819068, 0.232...|      5|
|"I have a version...|                  10|[0, 3, 5, 10, 18,...| [the, a, of, that...|[0.1819068, 0.285...|      4|
+--------------------+--------------------+--------------------+---------------------+--------------------+-------+
'''

# Output the file in JSON format because arrays cannot be accepted by csv unless converted to a string
final_outputs.write.format('json').save(f'hdfs://{argv[1]}:9000/reviews_tfidf.json')
