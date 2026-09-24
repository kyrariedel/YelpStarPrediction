Yelp Star Rating Regression (Spark MLlib)
=========================================

Objective: Predict a Yelp review's star rating from review text using
Spark MLlib. One run trains Decision Tree and Random Forest regressors (RMSE)
and weighted classifiers (accuracy).

v3 text features: lowercase, English stopwords, unigrams+bigrams, CountVectorizer
(default vocab 65536) + IDF. Classifiers use inverse-frequency class weights.
Metrics: RMSE/MAE, exact rounded accuracy, ±1-star accuracy, 3-class buckets.

Installation
------------
Tech Stack:
- OpenJDK 11
- Hadoop 3.3.5
- Maven 3.9.x
- AWS CLI (for EMR)
- Scala 2.12.x
- Spark 3.3.2 (without bundled Hadoop)

Edit paths at the top of the `Makefile` (`spark.root`, `hadoop.root`, AWS bucket).

Build & run (local)
-------------------
1. Create review subsets (1k / 10k / 100k / 1m) from the full Yelp JSONL:

   `make subsets`

2. Build and run the default 1k local job:

   `make local`

3. Scaling ladder:

   `make local-1k`
   `make local-10k`
   `make local-100k`
   `make local-1m`

Optional knobs (Makefile): `num.features`, `train.fraction`, `max.depth`,
`num.trees`, `seed`, `driver.memory`, `executor.memory`.

Program args
------------
`yelp.StarRatingMain <input> <output> [numFeatures] [trainFraction] [maxDepth] [numTrees] [seed]`

AWS EMR
-------
Configure `aws.*` in the Makefile, then:
- `make make-bucket`
- upload a chosen subset to `s3://$bucket/input` (or adapt `upload-input-aws`)
- `make aws`
- `make download-output-aws`
