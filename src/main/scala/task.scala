// DSCI 553 | Foundations and Applications of Data Mining
// Homework 6
// Matheus Schmitz
// USC ID: 5039286453

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg._
import scala.collection.mutable
import java.io._
import scala.collection.immutable.ListMap
import scala.io.Source
import scala.util.Random
import scala.util.control.NonFatal
import scala.util.Try


object task {

  def point_cluster_mahalanobis_distance(datapoint_idx: Int,
                                         cluster_stats: mutable.Map[String, mutable.ListBuffer[Double]],
                                         dataset_dimensionality: Int,
                                         dataset_dict: scala.collection.Map[Int, (Int, Array[Double])]): Double = {

    val stdev = cluster_stats("stdev")
    val centroid = cluster_stats("centroid")
    var mahalanobis_distance = 0.toDouble
    for (dim <- 0 until dataset_dimensionality) {
      mahalanobis_distance += math.pow(((dataset_dict(datapoint_idx)._2(dim) - centroid(dim)) / stdev(dim)), 2).toDouble
    }
    mahalanobis_distance = math.sqrt(mahalanobis_distance).toDouble

    return mahalanobis_distance
  }

  def intercluster_mahalanobis_distance(left_cluster_stats: mutable.Map[String, mutable.ListBuffer[Double]],
                                        right_cluster_stats: mutable.Map[String, mutable.ListBuffer[Double]],
                                        dataset_dimensionality: Int,
                                        dataset_dict: scala.collection.Map[Int, (Int, Array[Double])]): Double = {

    val (stdev1, centroid1) = (left_cluster_stats("stdev"), left_cluster_stats("centroid"))
    val (stdev2, centroid2) = (right_cluster_stats("stdev"), right_cluster_stats("centroid"))

    // Calculate the mahalanobis distance between the clusters
    var (left_cluster_dist, right_cluster_dist) = (0.toDouble, 0.toDouble)
    for (dim <- 0 until dataset_dimensionality) {
      left_cluster_dist += math.pow(((centroid1(dim) - centroid2(dim)) / stdev2(dim)), 2).toDouble
      right_cluster_dist += math.pow(((centroid2(dim) - centroid1(dim)) / stdev1(dim)), 2).toDouble
    }
    left_cluster_dist = math.sqrt(left_cluster_dist).toDouble
    right_cluster_dist = math.sqrt(right_cluster_dist).toDouble
    val min_dist = left_cluster_dist.min(right_cluster_dist)

    return min_dist
  }

//  def update_DS(datapoint_idx: Int,
//                cluster_idx: Int) = {
//
//    DISCARD_SET_stats(cluster_idx)("ids_in_cluster").append(datapoint_idx)
//    DISCARD_SET_stats(cluster_idx)("N")(0) += 1
//    for (dim <- 0 until dataset_dimensionality) {
//      DISCARD_SET_stats(cluster_idx)("sum")(dim) += dataset_dict(datapoint_idx)._2(dim)
//      DISCARD_SET_stats(cluster_idx)("sumsq")(dim) += math.pow(dataset_dict(datapoint_idx)._2(dim), 2).toDouble
//      DISCARD_SET_stats(cluster_idx)("stdev")(dim) = (DISCARD_SET_stats(cluster_idx)("sumsq")(dim) / DISCARD_SET_stats(cluster_idx)("N")(0) -
//        math.pow((DISCARD_SET_stats(cluster_idx)("sum")(dim) / DISCARD_SET_stats(cluster_idx)("N")(0)), 2)).toDouble
//      DISCARD_SET_stats(cluster_idx)("centroid")(dim) = DISCARD_SET_stats(cluster_idx)("sum")(dim) / DISCARD_SET_stats(cluster_idx)("N")(0)
//    }
//    return DISCARD_SET_stats
//  }
//
//  def update_CS(datapoint_idx: Int,
//                cluster_idx: Int) = {
//
//    COMPRESSION_SET_stats(cluster_idx)("ids_in_cluster").append(datapoint_idx)
//    COMPRESSION_SET_stats(cluster_idx)("N")(0) += 1
//    for (dim <- 0 until dataset_dimensionality) {
//      COMPRESSION_SET_stats(cluster_idx)("sum")(dim) += dataset_dict(datapoint_idx)._2(dim)
//      COMPRESSION_SET_stats(cluster_idx)("sumsq")(dim) += math.pow(dataset_dict(datapoint_idx)._2(dim), 2).toDouble
//      COMPRESSION_SET_stats(cluster_idx)("stdev")(dim) = (COMPRESSION_SET_stats(cluster_idx)("sumsq")(dim) / COMPRESSION_SET_stats(cluster_idx)("N")(0) -
//        math.pow((COMPRESSION_SET_stats(cluster_idx)("sum")(dim) / COMPRESSION_SET_stats(cluster_idx)("N")(0)), 2)).toDouble
//      COMPRESSION_SET_stats(cluster_idx)("centroid")(dim) = COMPRESSION_SET_stats(cluster_idx)("sum")(dim) / COMPRESSION_SET_stats(cluster_idx)("N")(0)
//    }
//    return COMPRESSION_SET_stats
//  }
//
//  def merge_CS_CS(key1: Int,
//                  key2: Int): mutable.Map[Int, mutable.Map[String, mutable.ListBuffer[Double]]] = {
//
//    COMPRESSION_SET_stats(key1)("ids_in_cluster") ++= COMPRESSION_SET_stats(key2)("ids_in_cluster")
//    COMPRESSION_SET_stats(key1)("N")(0) = COMPRESSION_SET_stats(key1)("N")(0) + COMPRESSION_SET_stats(key2)("N")(0)
//    for (dim <- 0 until dataset_dimensionality) {
//      COMPRESSION_SET_stats(key1)("sum")(dim) += COMPRESSION_SET_stats(key2)("sum")(dim)
//      COMPRESSION_SET_stats(key1)("sumsq")(dim) += math.pow(COMPRESSION_SET_stats(key2)("sum")(dim), 2).toDouble
//      COMPRESSION_SET_stats(key1)("stdev")(dim) = (COMPRESSION_SET_stats(key1)("sumsq")(dim) / COMPRESSION_SET_stats(key1)("N")(0) -
//        math.pow((COMPRESSION_SET_stats(key1)("sum")(dim) / COMPRESSION_SET_stats(key1)("N")(0)), 2)).toDouble
//      COMPRESSION_SET_stats(key1)("centroid")(dim) = COMPRESSION_SET_stats(key1)("sum")(dim) / COMPRESSION_SET_stats(key1)("N")(0)
//    }
//    // Remove the merged cluster
//    COMPRESSION_SET_stats -= key2
//
//    return COMPRESSION_SET_stats
//  }
//
//  def merge_CS_DS(keyCS: Int,
//                  keyDS: Int): (mutable.Map[Int, mutable.Map[String, mutable.ListBuffer[Double]]], mutable.Map[Int, mutable.Map[String, mutable.ListBuffer[Double]]]) = {
//
//    DISCARD_SET_stats(keyDS)("ids_in_cluster") ++= COMPRESSION_SET_stats(keyCS)("ids_in_cluster")
//    DISCARD_SET_stats(keyDS)("N")(0) = COMPRESSION_SET_stats(keyCS)("N")(0) + COMPRESSION_SET_stats(keyCS)("N")(0)
//    for (dim <- 0 until dataset_dimensionality) {
//      DISCARD_SET_stats(keyDS)("sum")(dim) += COMPRESSION_SET_stats(keyCS)("sum")(dim)
//      DISCARD_SET_stats(keyDS)("sumsq")(dim) += math.pow(COMPRESSION_SET_stats(keyCS)("sum")(dim), 2).toDouble
//      DISCARD_SET_stats(keyDS)("stdev")(dim) = (DISCARD_SET_stats(keyDS)("sumsq")(dim) / DISCARD_SET_stats(keyDS)("N")(0) -
//        math.pow((DISCARD_SET_stats(keyDS)("sum")(dim) / DISCARD_SET_stats(keyDS)("N")(0)), 2)).toDouble
//      DISCARD_SET_stats(keyDS)("centroid")(dim) = DISCARD_SET_stats(keyDS)("sum")(dim) / DISCARD_SET_stats(keyDS)("N")(0)
//    }
//    // Remove the merged cluster
//    COMPRESSION_SET_stats -= keyCS
//
//    return (COMPRESSION_SET_stats, DISCARD_SET_stats)
//  }

  def main(args: Array[String]): Unit = {

    val start_time = System.currentTimeMillis()

    // Read user inputs
    val input_file = args(0)
    val n_cluster = args(1).toInt
    val output_file = args(2)
    //val input_file = "publicdata/hw6_clustering_mini.txt"
    //val n_cluster = 10
    //val output_file = "output_scala.txt"

    // Initialize Spark with the 4 GB memory parameters from HW1
    val config = new SparkConf().setMaster("local[*]").setAppName("task").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")//.set("spark.testing.memory", "471859200")
    val sc = SparkContext.getOrCreate(config)
    sc.setLogLevel("ERROR")

    // Convert the dataset to a dict for easily accessing the relevant information from a sample's id
    val dataset_dict = sc.textFile(input_file)
      .map(row => (row.split(',')(0).toInt, (row.split(',')(1).toInt, row.split(',').drop(2).map(_.toDouble))))
      .collectAsMap()

    // List all sample_ids which have not yet been clustered
    val unused_ids = dataset_dict.keys.toSet.to[collection.mutable.Set]

    // From the file figure out the sample size for each of the 5 iterations
    val sample_size = math.ceil(dataset_dict.size * 0.2).toInt

    // Get the dataset's dimensionality
    val dataset_dimensionality = dataset_dict(0)._2.length

    // Calculate the Mahalanobis threshold to be used
    val mahalanobis_threshold = 2 * math.sqrt(dataset_dimensionality)

    // Initialize the objects for the different sets
    var DS_CLUSTERS: mutable.Map[Int, mutable.ListBuffer[Int]] = mutable.Map()
    var DISCARD_SET_stats: mutable.Map[Int, mutable.Map[String, mutable.ListBuffer[Double]]] = mutable.Map()
    var COMPRESSION_SET_stats: mutable.Map[Int, mutable.Map[String, mutable.ListBuffer[Double]]] = mutable.Map()
    var RETAINED_SET: mutable.ListBuffer[Int] = mutable.ListBuffer()

    // Sample the first round of data to initialize the algorithm
    var round_1_samples = Random.shuffle(unused_ids).take(sample_size)
    for (sample_id <- round_1_samples) {
      unused_ids -= sample_id
    }

    // Create the K-Means training data from the round_1_samples
    var init_sample_features = mutable.ListBuffer.empty[Vector]
    for (sample_id <- round_1_samples){
      init_sample_features.append(Vectors.dense(dataset_dict(sample_id)._2.map(_.toDouble)))
    }
    var X_train = sc.parallelize(init_sample_features)

    // Run K-Means (e.g., from sklearn) with a large K (e.g., 5 times of the number of the input clusters)
    var kmeans = new KMeans().setK(20*n_cluster).setInitializationSteps(100).setMaxIterations(1000)
    var kmeans_trained = kmeans.run(X_train)
    var predicted_clusters = kmeans_trained.predict(X_train).collect()

    // Assign points to clusters
    var init_clusters: mutable.Map[Int, mutable.ListBuffer[Int]] = mutable.Map()
    for ((cluster_id, sampled_id) <- predicted_clusters.zip(round_1_samples)) {
      // If there is no key for the current cluster id, create one
      if (!init_clusters.contains(cluster_id)) {
        init_clusters += cluster_id -> mutable.ListBuffer()
      }
      init_clusters(cluster_id).append(sampled_id)
    }

    // Move lone points to the RETAINED_SET
    for ((cluster_id, clustered_points) <- init_clusters) {
      if (clustered_points.length == 1) {
        RETAINED_SET.append(clustered_points(0))
        round_1_samples -= clustered_points(0)
      }
    }

    // Cluster the round_1_samples after having moved outliers to the RS
    var train_sample_features = mutable.ListBuffer.empty[Vector]
    for (sample_id <- round_1_samples){
      train_sample_features.append(Vectors.dense(dataset_dict(sample_id)._2.map(_.toDouble)))
    }
    X_train = sc.parallelize(train_sample_features)
    kmeans = new KMeans().setK(n_cluster).setInitializationSteps(100).setMaxIterations(1000)
    kmeans_trained = kmeans.run(X_train)
    predicted_clusters = kmeans_trained.predict(X_train).collect()

    // Assign points to DS clusters
    for ((cluster_id, sampled_id) <- predicted_clusters.zip(round_1_samples)) {
      // If there is no key for the current cluster id, create one
      if (!DS_CLUSTERS.contains(cluster_id)) {
        DS_CLUSTERS += cluster_id -> mutable.ListBuffer()
      }
      DS_CLUSTERS(cluster_id).append(sampled_id)
    }

    // Calculate the cluster statistics for the clusters generated from the initialization round
    var features_matrix = mutable.ListBuffer.empty[mutable.ListBuffer[Double]]
    for((key, value) <- DS_CLUSTERS) {
      DISCARD_SET_stats += key -> mutable.Map()
      DISCARD_SET_stats(key) += ("ids_in_cluster" -> mutable.ListBuffer.empty[Double],
                                "N" -> mutable.ListBuffer.empty[Double],
                                "sum" -> mutable.ListBuffer.empty[Double],
                                "sumsq" -> mutable.ListBuffer.empty[Double],
                                "stdev" -> mutable.ListBuffer.empty[Double],
                                "centroid" -> mutable.ListBuffer.empty[Double])
      // Populate the ListBuffers with placeholder for the column values (each of the dimensions)
      for (dim <- 0 until dataset_dimensionality){
        DISCARD_SET_stats(key)("N").append(0.toDouble)
        DISCARD_SET_stats(key)("sum").append(0.toDouble)
        DISCARD_SET_stats(key)("sumsq").append(0.toDouble)
        DISCARD_SET_stats(key)("stdev").append(0.toDouble)
        DISCARD_SET_stats(key)("centroid").append(0.toDouble)
      }
      features_matrix = mutable.ListBuffer.empty[mutable.ListBuffer[Double]]
      for (datapoint <- value) {
        DISCARD_SET_stats(key)("ids_in_cluster").append(datapoint)
        features_matrix.append(dataset_dict(datapoint)._2.to[mutable.ListBuffer])
      }
      DISCARD_SET_stats(key)("N")(0) = DISCARD_SET_stats(key)("ids_in_cluster").length.toDouble
      for (sample_features <- features_matrix) {
        for (dim <- 0 until dataset_dimensionality) {
          DISCARD_SET_stats(key)("sum")(dim) += sample_features(dim)
          DISCARD_SET_stats(key)("sumsq")(dim) += math.pow(sample_features(dim),2).toDouble
        }
      }
      for (dim <- 0 until dataset_dimensionality) {
        DISCARD_SET_stats(key)("stdev")(dim) = math.sqrt(DISCARD_SET_stats(key)("sumsq")(dim)/DISCARD_SET_stats(key)("N")(0) -
                                                          math.pow((DISCARD_SET_stats(key)("sum")(dim)/DISCARD_SET_stats(key)("N")(0)), 2)).toDouble
        DISCARD_SET_stats(key)("centroid")(dim) = DISCARD_SET_stats(key)("sum")(dim)/DISCARD_SET_stats(key)("N")(0)
      }
    }

    // Run KMeans on the RETAINED_SET and generate COMPRESSION_SET clusters
    var RS_sample_features = mutable.ListBuffer.empty[Vector]
    for (sample_id <- RETAINED_SET){
      RS_sample_features.append(Vectors.dense(dataset_dict(sample_id)._2.map(_.toDouble)))
    }
    X_train = sc.parallelize(RS_sample_features)
    // If there were any samples in the RS, cluster them
    val RS_clusters = mutable.Map.empty[Int, mutable.ListBuffer[Int]]
    if (RETAINED_SET.nonEmpty) {
      kmeans = new KMeans().setK((7*n_cluster).min(math.floor(1 + (RETAINED_SET.length/2)).toInt)).setInitializationSteps(100).setMaxIterations(1000)
      kmeans_trained = kmeans.run(X_train)
      predicted_clusters = kmeans_trained.predict(X_train).collect()
      for ((cluster_id, sampled_id) <- predicted_clusters.zip(RETAINED_SET)) {
        // If there is no key for the current cluster id, create one
        if (!RS_clusters.contains(cluster_id)) {
          RS_clusters += cluster_id -> mutable.ListBuffer()
        }
        RS_clusters(cluster_id).append(sampled_id)
      }
    }

    // Calculate the cluster statistics for the clusters generated from the initialization round
    features_matrix = mutable.ListBuffer.empty[mutable.ListBuffer[Double]]
    for((key, value) <- RS_clusters) {
      COMPRESSION_SET_stats += key -> mutable.Map()
      COMPRESSION_SET_stats(key) += ("ids_in_cluster" -> mutable.ListBuffer.empty[Double],
        "N" -> mutable.ListBuffer.empty[Double],
        "sum" -> mutable.ListBuffer.empty[Double],
        "sumsq" -> mutable.ListBuffer.empty[Double],
        "stdev" -> mutable.ListBuffer.empty[Double],
        "centroid" -> mutable.ListBuffer.empty[Double])
      // Populate the ListBuffers with placeholder for the column values (each of the dimensions)
      for (dim <- 0 until dataset_dimensionality){
        COMPRESSION_SET_stats(key)("N").append(0.toDouble)
        COMPRESSION_SET_stats(key)("sum").append(0.toDouble)
        COMPRESSION_SET_stats(key)("sumsq").append(0.toDouble)
        COMPRESSION_SET_stats(key)("stdev").append(0.toDouble)
        COMPRESSION_SET_stats(key)("centroid").append(0.toDouble)
      }
      features_matrix = mutable.ListBuffer.empty[mutable.ListBuffer[Double]]
      for (datapoint <- value) {
        COMPRESSION_SET_stats(key)("ids_in_cluster").append(datapoint)
        features_matrix.append(dataset_dict(datapoint)._2.to[mutable.ListBuffer])
      }
      COMPRESSION_SET_stats(key)("N")(0) = COMPRESSION_SET_stats(key)("ids_in_cluster").length.toDouble
      for (sample_features <- features_matrix) {
        for (dim <- 0 until dataset_dimensionality) {
          COMPRESSION_SET_stats(key)("sum")(dim) += sample_features(dim)
          COMPRESSION_SET_stats(key)("sumsq")(dim) += math.pow(sample_features(dim),2).toDouble
        }
      }
      for (dim <- 0 until dataset_dimensionality) {
        COMPRESSION_SET_stats(key)("stdev")(dim) = math.sqrt(COMPRESSION_SET_stats(key)("sumsq")(dim) / COMPRESSION_SET_stats(key)("N")(0) -
                                                             math.pow((COMPRESSION_SET_stats(key)("sum")(dim) / COMPRESSION_SET_stats(key)("N")(0)), 2)).toDouble
        COMPRESSION_SET_stats(key)("centroid")(dim) = COMPRESSION_SET_stats(key)("sum")(dim) / COMPRESSION_SET_stats(key)("N")(0)
      }

      // Clean the samples from the retained set which have been summarized
      for (datapoint <- value) {
        RETAINED_SET -= datapoint
      }
    }

    // Objects to tally values to be outputted
    var num_DS_points = 0
    var num_CS_clusters = 0
    var num_CS_points = 0
    var num_RS_points = 0

    // Tally the values for outputting
    for ((key, value) <- DISCARD_SET_stats) {
      num_DS_points += value("N")(0).toInt
    }
    for ((key, value) <- COMPRESSION_SET_stats) {
      num_CS_clusters += 1
      num_CS_points += value("N")(0).toInt
    }
    num_RS_points = RETAINED_SET.length

    // Write the results for the initialization round (round 1)
    val pw = new PrintWriter(new File(output_file))
    pw.write("The intermediate results:" + "\n")
    pw.write("Round 1: " + num_DS_points.toString + "," + num_CS_clusters.toString + "," + num_CS_points.toString + "," + num_RS_points.toString)

//    println("Round 1 Stats:")
//    for (key <- DISCARD_SET_stats.keys.toList.sorted) {
//      println("Cluster: " + key.toString + ". Size: " + DISCARD_SET_stats(key)("N")(0).toString)
//    }

    // ### Initialization is finished, run the regular iterations ###
    var iteration_samples: mutable.Set[Int] = mutable.Set.empty[Int]
    for (curr_round <- 2 until 6) {

      iteration_samples = mutable.Set.empty[Int]
      // Load the samples for the iteration about the begin
      if (curr_round < 5) {
        iteration_samples = Random.shuffle(unused_ids).take(sample_size)
        for (sample_id <- iteration_samples) {
          unused_ids -= sample_id
        }
      }
      else {
        iteration_samples ++= unused_ids
        for (sample_id <- iteration_samples) {
          unused_ids -= sample_id
        }
      }

      // ##################################
      // ### Assign Samples to DS/CS/RS ###
      // ##################################

      // Iterate over each sample among those drawn for this iterations
      for (sample_id <- iteration_samples) {

        // Default to assuming points are in the RETAINED_SET
        var assigned_cluster = -1

        // Track the lowest distance between the sample and all clusters
        var lowest_dist = mahalanobis_threshold

        // Find the DISCARD_SET cluster closest to the current sample
        for ((cluster_id, cluster_stats) <- DISCARD_SET_stats) {

          var mahalanobis_distance = point_cluster_mahalanobis_distance(sample_id, cluster_stats, dataset_dimensionality, dataset_dict)

          // If the distance is under the mahalanobis_threshold and also the lowest distance yet found, update the point's cluster
          if (mahalanobis_distance < lowest_dist) {
            assigned_cluster = cluster_id
            lowest_dist = mahalanobis_distance
          }
        }

        // Update the statistics of the cluster to which the point is assigned
        if (assigned_cluster != -1) {
          DISCARD_SET_stats(assigned_cluster)("ids_in_cluster").append(sample_id)
          DISCARD_SET_stats(assigned_cluster)("N")(0) += 1
          for (dim <- 0 until dataset_dimensionality) {
            DISCARD_SET_stats(assigned_cluster)("sum")(dim) += dataset_dict(sample_id)._2(dim)
            DISCARD_SET_stats(assigned_cluster)("sumsq")(dim) += math.pow(dataset_dict(sample_id)._2(dim), 2).toDouble
            DISCARD_SET_stats(assigned_cluster)("stdev")(dim) = math.sqrt(DISCARD_SET_stats(assigned_cluster)("sumsq")(dim) / DISCARD_SET_stats(assigned_cluster)("N")(0) -
                                                                            math.pow((DISCARD_SET_stats(assigned_cluster)("sum")(dim) / DISCARD_SET_stats(assigned_cluster)("N")(0)), 2)).toDouble
            DISCARD_SET_stats(assigned_cluster)("centroid")(dim) = DISCARD_SET_stats(assigned_cluster)("sum")(dim) / DISCARD_SET_stats(assigned_cluster)("N")(0)
          }
        }

        // If the sample could not be assigned to any cluster in the DS, try assigning it to the CS
        else {

          // Find the COMPRESSION_SET cluster closest to the current sample
          for ((cluster_id, cluster_stats) <- COMPRESSION_SET_stats) {

            var mahalanobis_distance = point_cluster_mahalanobis_distance(sample_id, cluster_stats, dataset_dimensionality, dataset_dict)

            // If the distance is under the mahalanobis_threshold and also the lowest distance yet found, update the point's cluster
            if (mahalanobis_distance < lowest_dist) {
              assigned_cluster = cluster_id
              lowest_dist = mahalanobis_distance
            }
          }
          // Update the statistics of the cluster to which the point is assigned
          if (assigned_cluster != -1) {
            COMPRESSION_SET_stats(assigned_cluster)("ids_in_cluster").append(sample_id)
            COMPRESSION_SET_stats(assigned_cluster)("N")(0) += 1
            for (dim <- 0 until dataset_dimensionality) {
              COMPRESSION_SET_stats(assigned_cluster)("sum")(dim) += dataset_dict(sample_id)._2(dim)
              COMPRESSION_SET_stats(assigned_cluster)("sumsq")(dim) += math.pow(dataset_dict(sample_id)._2(dim), 2).toDouble
              COMPRESSION_SET_stats(assigned_cluster)("stdev")(dim) = math.sqrt(COMPRESSION_SET_stats(assigned_cluster)("sumsq")(dim) / COMPRESSION_SET_stats(assigned_cluster)("N")(0) -
                                                                                math.pow((COMPRESSION_SET_stats(assigned_cluster)("sum")(dim) / COMPRESSION_SET_stats(assigned_cluster)("N")(0)), 2)).toDouble
              COMPRESSION_SET_stats(assigned_cluster)("centroid")(dim) = COMPRESSION_SET_stats(assigned_cluster)("sum")(dim) / COMPRESSION_SET_stats(assigned_cluster)("N")(0)
            }
          }

          // If the BRF also failed to assing the sample to a CS cluster, send it to the RS
          else {
            RETAINED_SET.append(sample_id)
          }
        }
      }

      // ######################################
      // ### Create New CSs from RS Samples ###
      // ######################################

      // Run KMeans on the RETAINED_SET and generate COMPRESSION_SET clusters
      var RS_sample_features = mutable.ListBuffer.empty[Vector]
      for (sample_id <- RETAINED_SET) {
        RS_sample_features.append(Vectors.dense(dataset_dict(sample_id)._2.map(_.toDouble)))
      }
      X_train = sc.parallelize(RS_sample_features)
      // If there were any samples in the RS, cluster them
      val RS_clusters = mutable.Map.empty[Int, mutable.ListBuffer[Int]]
      if (RETAINED_SET.nonEmpty) {
        kmeans = new KMeans().setK((7*n_cluster).min(math.floor(1 + (RETAINED_SET.length / 2)).toInt)).setInitializationSteps(100).setMaxIterations(1000)
        kmeans_trained = kmeans.run(X_train)
        predicted_clusters = kmeans_trained.predict(X_train).collect()
        for ((cluster_id, sampled_id) <- predicted_clusters.zip(RETAINED_SET)) {
          // If there is no key for the current cluster id, create one
          if (!RS_clusters.contains(cluster_id)) {
            RS_clusters += cluster_id -> mutable.ListBuffer()
          }
          RS_clusters(cluster_id).append(sampled_id)
        }
      }

      // Calculate the cluster statistics for the clusters generated from the initialization round
      features_matrix = mutable.ListBuffer.empty[mutable.ListBuffer[Double]]
      for ((key, value) <- RS_clusters) {

        // Only clusters with more than one sample in them go to the CS, those with only 1 sample remain RS
        if (value.length > 1) {

          // Find the find the next cluster index to use for the COMPRESSION_SET_stats
          var CS_stats_next_key = 0
          Try {
            CS_stats_next_key = COMPRESSION_SET_stats.keys.max + 1
          }

          COMPRESSION_SET_stats += CS_stats_next_key -> mutable.Map()
          COMPRESSION_SET_stats(CS_stats_next_key) += ("ids_in_cluster" -> mutable.ListBuffer.empty[Double],
            "N" -> mutable.ListBuffer.empty[Double],
            "sum" -> mutable.ListBuffer.empty[Double],
            "sumsq" -> mutable.ListBuffer.empty[Double],
            "stdev" -> mutable.ListBuffer.empty[Double],
            "centroid" -> mutable.ListBuffer.empty[Double])
          // Populate the ListBuffers with placeholder for the column values (each of the dimensions)
          for (dim <- 0 until dataset_dimensionality) {
            COMPRESSION_SET_stats(CS_stats_next_key)("N").append(0.toDouble)
            COMPRESSION_SET_stats(CS_stats_next_key)("sum").append(0.toDouble)
            COMPRESSION_SET_stats(CS_stats_next_key)("sumsq").append(0.toDouble)
            COMPRESSION_SET_stats(CS_stats_next_key)("stdev").append(0.toDouble)
            COMPRESSION_SET_stats(CS_stats_next_key)("centroid").append(0.toDouble)
          }
          features_matrix = mutable.ListBuffer.empty[mutable.ListBuffer[Double]]
          for (datapoint <- value) {
            COMPRESSION_SET_stats(CS_stats_next_key)("ids_in_cluster").append(datapoint)
            features_matrix.append(dataset_dict(datapoint)._2.to[mutable.ListBuffer])
          }
          COMPRESSION_SET_stats(CS_stats_next_key)("N")(0) = COMPRESSION_SET_stats(CS_stats_next_key)("ids_in_cluster").length.toDouble
          for (sample_features <- features_matrix) {
            for (dim <- 0 until dataset_dimensionality) {
              COMPRESSION_SET_stats(CS_stats_next_key)("sum")(dim) += sample_features(dim)
              COMPRESSION_SET_stats(CS_stats_next_key)("sumsq")(dim) += math.pow(sample_features(dim), 2).toDouble
            }
          }
          for (dim <- 0 until dataset_dimensionality) {
            COMPRESSION_SET_stats(CS_stats_next_key)("stdev")(dim) = math.sqrt(COMPRESSION_SET_stats(CS_stats_next_key)("sumsq")(dim) / COMPRESSION_SET_stats(CS_stats_next_key)("N")(0) -
                                                                     math.pow((COMPRESSION_SET_stats(CS_stats_next_key)("sum")(dim) / COMPRESSION_SET_stats(CS_stats_next_key)("N")(0)), 2)).toDouble
            COMPRESSION_SET_stats(CS_stats_next_key)("centroid")(dim) = COMPRESSION_SET_stats(CS_stats_next_key)("sum")(dim) / COMPRESSION_SET_stats(CS_stats_next_key)("N")(0)
          }

          // Clean the samples from the retained set which have been summarized
          for (datapoint <- value) {
            RETAINED_SET -= datapoint
          }
        }
      }

      // #################################################
      // ### Merge CSs Below the Mahalanobis Threshold ###
      // #################################################

      var close_CSs: mutable.Map[Int, Int] = mutable.Map.empty[Int, Int]
      for ((key1, value1) <- COMPRESSION_SET_stats) {

        // Default to assuming there is no other CS cluster close by
        var assigned_cluster: Int = -1

        // Track the lowest distance between the sample and all clusters
        var lowest_dist = mahalanobis_threshold

        // Compare each CS cluster to all other CS clusters
        for ((key2, value2) <- COMPRESSION_SET_stats) {

          // Do not compare the a cluster to itself
          if (key1 != key2) {

            var intercluster_dist = intercluster_mahalanobis_distance(value1, value2, dataset_dimensionality, dataset_dict)

            // If the intercluster distance is below the threshold, make the pair a candidate for merging
            if (intercluster_dist < lowest_dist) {
              assigned_cluster = key2
              lowest_dist = intercluster_dist
            }
          }
        }
        // Once all pairwise comparisons were done for a given cluster (key1) store the results
        close_CSs += key1 -> assigned_cluster
      }
      // Once all closest CS clusters were found, merge them and update the COMPRESSION_SET_stats
      for ((key1, key2) <- close_CSs) {
        if (COMPRESSION_SET_stats.contains(key1) && COMPRESSION_SET_stats.contains(key2) && key1 != key2) {
          COMPRESSION_SET_stats(key1)("ids_in_cluster") ++= COMPRESSION_SET_stats(key2)("ids_in_cluster")
          COMPRESSION_SET_stats(key1)("N")(0) += COMPRESSION_SET_stats(key2)("N")(0)
          for (dim <- 0 until dataset_dimensionality) {
            COMPRESSION_SET_stats(key1)("sum")(dim) += COMPRESSION_SET_stats(key2)("sum")(dim)
            COMPRESSION_SET_stats(key1)("sumsq")(dim) += math.pow(COMPRESSION_SET_stats(key2)("sum")(dim), 2).toDouble
            COMPRESSION_SET_stats(key1)("stdev")(dim) = math.sqrt(COMPRESSION_SET_stats(key1)("sumsq")(dim) / COMPRESSION_SET_stats(key1)("N")(0) -
                                                                   math.pow((COMPRESSION_SET_stats(key1)("sum")(dim) / COMPRESSION_SET_stats(key1)("N")(0)), 2)).toDouble
            COMPRESSION_SET_stats(key1)("centroid")(dim) = COMPRESSION_SET_stats(key1)("sum")(dim) / COMPRESSION_SET_stats(key1)("N")(0)
          }
          // Remove the merged cluster
          COMPRESSION_SET_stats -= key2
        }
      }

      // ######################################################
      // ### At the Final Iteration Merge CSs to Nearby DSs ###
      // ######################################################

      // Check if it is the last iteration
      if (curr_round == 5) {

        var close_DSs: mutable.Map[Int, Int] = mutable.Map.empty[Int, Int]
        for ((cs_key, cs_value) <- COMPRESSION_SET_stats) {

          // Default to assuming there is no other CS cluster close by
          var assigned_cluster: Int = -1

          // Track the lowest distance between the sample and all clusters
          var lowest_dist = mahalanobis_threshold

          // Compare each CS cluster to all other CS clusters
          for ((ds_key, ds_value) <- DISCARD_SET_stats) {

            // Do not compare the a cluster to itself
            if (cs_key != ds_key) {

              var intercluster_dist = intercluster_mahalanobis_distance(cs_value, ds_value, dataset_dimensionality, dataset_dict)

              // If the intercluster distance is below the threshold, make the pair a candidate for merging
              if (intercluster_dist < lowest_dist) {
                assigned_cluster = ds_key
                lowest_dist = intercluster_dist
              }
            }
          }
          // Once all pairwise comparisons were done for a given cluster (key1) store the results
          close_DSs += cs_key -> assigned_cluster
        }
        // Once all closest CS clusters were found, merge them and update the COMPRESSION_SET_stats
        for ((keyCS, keyDS) <- close_DSs) {
          if (COMPRESSION_SET_stats.contains(keyCS) && DISCARD_SET_stats.contains(keyDS) && keyCS != keyDS) {
            DISCARD_SET_stats(keyDS)("ids_in_cluster") ++= COMPRESSION_SET_stats(keyCS)("ids_in_cluster")
            DISCARD_SET_stats(keyDS)("N")(0) += COMPRESSION_SET_stats(keyCS)("N")(0)
            for (dim <- 0 until dataset_dimensionality) {
              DISCARD_SET_stats(keyDS)("sum")(dim) += COMPRESSION_SET_stats(keyCS)("sum")(dim)
              DISCARD_SET_stats(keyDS)("sumsq")(dim) += math.pow(COMPRESSION_SET_stats(keyCS)("sum")(dim), 2).toDouble
              DISCARD_SET_stats(keyDS)("stdev")(dim) = math.sqrt(DISCARD_SET_stats(keyDS)("sumsq")(dim) / DISCARD_SET_stats(keyDS)("N")(0) -
                                                                  math.pow((DISCARD_SET_stats(keyDS)("sum")(dim) / DISCARD_SET_stats(keyDS)("N")(0)), 2)).toDouble
              DISCARD_SET_stats(keyDS)("centroid")(dim) = DISCARD_SET_stats(keyDS)("sum")(dim) / DISCARD_SET_stats(keyDS)("N")(0)
            }
            // Remove the merged cluster
            COMPRESSION_SET_stats -= keyCS
          }
        }
      }

      // ###############################
      // ### Output Round Statistics ###
      // ###############################

      // Objects to tally values to be outputted
      num_DS_points = 0
      num_CS_clusters = 0
      num_CS_points = 0
      num_RS_points = 0

      // Tally the values for outputting
      for ((key, value) <- DISCARD_SET_stats) {
        num_DS_points += value("N")(0).toInt
      }
      for ((key, value) <- COMPRESSION_SET_stats) {
        num_CS_clusters += 1
        num_CS_points += value("N")(0).toInt
      }
      num_RS_points = RETAINED_SET.length

      // Write the results for the initialization round (round 1)
      pw.write("\n" + "Round " + curr_round.toString + ": " + num_DS_points.toString + "," + num_CS_clusters.toString + "," + num_CS_points.toString + "," + num_RS_points.toString)


//      println("\n" + "Round " + curr_round.toString + " Stats: ")
//      for (key <- DISCARD_SET_stats.keys.toList.sorted) {
//        println("Cluster: " + key.toString + ". Size: " + DISCARD_SET_stats(key)("N")(0).toString)
//      }
    }

    // Go over all sets (DS, CS, RS) and extract each datapoint's final cluster
    var sample_id_to_cluster: mutable.Map[Int, Int] = mutable.Map.empty[Int, Int]
    for ((key, value) <- DISCARD_SET_stats) {
      for (datapoint <- value("ids_in_cluster")) {
        sample_id_to_cluster += datapoint.toInt -> key
      }
    }
    for ((key, value) <- COMPRESSION_SET_stats) {
      for (datapoint <- value("ids_in_cluster")) {
        sample_id_to_cluster += datapoint.toInt -> -1
      }
    }
    for (datapoint <- RETAINED_SET) {
      sample_id_to_cluster += datapoint.toInt -> -1
    }

    // Write the clustering results
    pw.write("\n" + "\n" + "The clustering results:")
    for (key <- sample_id_to_cluster.keys.toList.sorted) {
      pw.write("\n" + key.toString + "," + sample_id_to_cluster(key).toString)
    }

    // Stop Spark
    pw.close()
    sc.stop()

    // Measure the total time taken and report it
    val total_time = System.currentTimeMillis() - start_time
    val time_elapsed = total_time.toDouble / 1000.toDouble
    println("Duration: " + time_elapsed)
  }
}
