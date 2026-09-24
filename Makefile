# Makefile for Yelp star-rating Spark MLlib project.

# Customize these paths for your environment.
# -----------------------------------------------------------
spark.root=/usr/local/spark-3.3.2-bin-without-hadoop
hadoop.root=/usr/local/hadoop-3.3.5
spark.submit=$(spark.root)/bin/spark-submit
app.name=Yelp Star Rating
jar.name=yelp-stars.jar
maven.jar.name=yelp-stars-1.0.jar
job.name=yelp.StarRating
local.master=local[4]
local.input=input/reviews_1k.json
local.output=output
driver.memory=4g
executor.memory=4g
# Model / feature knobs (passed through to the job)
num.features=8192
train.fraction=0.8
max.depth=8
num.trees=50
seed=42
# Pseudo-Cluster Execution
hdfs.user.name=kyrariedel
hdfs.input=input
hdfs.output=output
# AWS EMR Execution
aws.emr.release=emr-6.10.0
aws.bucket.name=cs6240-demo-bucket-kr1
aws.input=input_1m
aws.output=output
aws.log.dir=log
aws.core.num.nodes=2
aws.primary.num.nodes=1
aws.instance.type=m5.xlarge
# -----------------------------------------------------------

# Compiles code and builds jar (with dependencies).
jar:
	mvn clean package
	cp target/${maven.jar.name} ${jar.name}

# Removes local output directory.
clean-local-output:
	rm -rf ${local.output}*

# Runs standalone (default: 1k reviews).
local: jar clean-local-output
	SPARK_DIST_CLASSPATH="$$(${hadoop.root}/bin/hadoop classpath)" \
	${spark.submit} --class ${job.name} --master ${local.master} --name "${app.name}" \
		--driver-memory ${driver.memory} --executor-memory ${executor.memory} \
		${jar.name} ${local.input} ${local.output} \
		${num.features} ${train.fraction} ${max.depth} ${num.trees} ${seed}

# Convenience targets for the local scaling ladder (v3 dirs; does not overwrite v2).
local-1k:
	$(MAKE) local local.input=input/reviews_1k.json local.output=output_v3_1k

local-10k:
	$(MAKE) local local.input=input/reviews_10k.json local.output=output_v3_10k

local-100k:
	$(MAKE) local local.input=input/reviews_100k.json local.output=output_v3_100k driver.memory=6g executor.memory=6g

local-1m:
	$(MAKE) local local.input=input/reviews_1m.json local.output=output_v3_1m driver.memory=8g executor.memory=8g

# Build line-count subsets from the full Yelp review JSONL.
subsets:
	python3 scripts/subset_data.py

# Start HDFS
start-hdfs:
	${hadoop.root}/sbin/start-dfs.sh

# Stop HDFS
stop-hdfs:
	${hadoop.root}/sbin/stop-dfs.sh

# Start YARN
start-yarn: stop-yarn
	${hadoop.root}/sbin/start-yarn.sh

# Stop YARN
stop-yarn:
	${hadoop.root}/sbin/stop-yarn.sh

# Reformats & initializes HDFS.
format-hdfs: stop-hdfs
	rm -rf /tmp/hadoop*
	${hadoop.root}/bin/hdfs namenode -format

# Initializes user & input directories of HDFS.
init-hdfs: start-hdfs
	${hadoop.root}/bin/hdfs dfs -rm -r -f /user
	${hadoop.root}/bin/hdfs dfs -mkdir /user
	${hadoop.root}/bin/hdfs dfs -mkdir /user/${hdfs.user.name}
	${hadoop.root}/bin/hdfs dfs -mkdir /user/${hdfs.user.name}/${hdfs.input}

# Load data to HDFS
upload-input-hdfs: start-hdfs
	${hadoop.root}/bin/hdfs dfs -put ${local.input} /user/${hdfs.user.name}/${hdfs.input}/

# Removes hdfs output directory.
clean-hdfs-output:
	${hadoop.root}/bin/hdfs dfs -rm -r -f ${hdfs.output}*

# Download output from HDFS to local.
download-output-hdfs:
	mkdir ${local.output}
	${hadoop.root}/bin/hdfs dfs -get ${hdfs.output}/* ${local.output}

# Runs pseudo-clustered (ALL). ONLY RUN THIS ONCE, THEN USE: make pseudoq
pseudo: jar stop-yarn format-hdfs init-hdfs upload-input-hdfs start-yarn clean-local-output
	${spark.submit} --class ${job.name} --master yarn --deploy-mode cluster ${jar.name} \
		/user/${hdfs.user.name}/${hdfs.input} ${hdfs.output} \
		${num.features} ${train.fraction} ${max.depth} ${num.trees} ${seed}
	make download-output-hdfs

# Runs pseudo-clustered (quickie).
pseudoq: jar clean-local-output clean-hdfs-output
	${spark.submit} --class ${job.name} --master yarn --deploy-mode cluster ${jar.name} \
		/user/${hdfs.user.name}/${hdfs.input} ${hdfs.output} \
		${num.features} ${train.fraction} ${max.depth} ${num.trees} ${seed}
	make download-output-hdfs

# Create S3 bucket.
make-bucket:
	aws s3 mb s3://${aws.bucket.name}

# Upload data to S3 input dir.
upload-input-aws: make-bucket
	aws s3 sync ${local.input} s3://${aws.bucket.name}/${aws.input}

# Delete S3 output dir.
delete-output-aws:
	aws s3 rm s3://${aws.bucket.name}/ --recursive --exclude "*" --include "${aws.output}*"

# Upload application to S3 bucket.
upload-app-aws:
	aws s3 cp ${jar.name} s3://${aws.bucket.name}

# Main EMR launch.
aws: jar upload-app-aws delete-output-aws
	aws emr create-cluster \
		--name "Yelp Star Rating Spark Cluster" \
		--release-label ${aws.emr.release} \
		--instance-groups '[{"InstanceCount":${aws.core.num.nodes},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":${aws.primary.num.nodes},"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
		--applications Name=Hadoop Name=Spark \
		--steps Type=CUSTOM_JAR,Name="${app.name}",Jar="command-runner.jar",ActionOnFailure=TERMINATE_CLUSTER,Args=["spark-submit","--deploy-mode","cluster","--class","${job.name}","s3://${aws.bucket.name}/${jar.name}","s3://${aws.bucket.name}/${aws.input}","s3://${aws.bucket.name}/${aws.output}","${num.features}","${train.fraction}","${max.depth}","${num.trees}","${seed}"] \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--configurations '[{"Classification": "hadoop-env", "Configurations": [{"Classification": "export","Configurations": [],"Properties": {"JAVA_HOME": "/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties": {}}, {"Classification": "spark-env", "Configurations": [{"Classification": "export","Configurations": [],"Properties": {"JAVA_HOME": "/usr/lib/jvm/java-11-amazon-corretto.x86_64"}}],"Properties": {}}]' \
		--use-default-roles \
		--enable-debugging \
		--auto-terminate

# Download output from S3.
download-output-aws: clean-local-output
	mkdir ${local.output}
	aws s3 sync s3://${aws.bucket.name}/${aws.output} ${local.output}
	aws s3 sync s3://${aws.bucket.name}/${aws.log.dir} log_aws

# Change to standalone mode.
switch-standalone:
	cp config/standalone/*.xml ${hadoop.root}/etc/hadoop

# Change to pseudo-cluster mode.
switch-pseudo:
	cp config/pseudo/*.xml ${hadoop.root}/etc/hadoop

# Package for release.
distro:
	rm -f Yelp-Stars.tar.gz
	rm -f Yelp-Stars.zip
	rm -rf build
	mkdir -p build/deliv/Yelp-Stars
	cp -r src build/deliv/Yelp-Stars
	cp -r config build/deliv/Yelp-Stars
	cp -r scripts build/deliv/Yelp-Stars
	cp pom.xml build/deliv/Yelp-Stars
	cp Makefile build/deliv/Yelp-Stars
	cp README.md build/deliv/Yelp-Stars
	tar -czf Yelp-Stars.tar.gz -C build/deliv Yelp-Stars
	cd build/deliv && zip -rq ../../Yelp-Stars.zip Yelp-Stars
