'''
DSCI 553 | Foundations and Applications of Data Mining
Homework 6
Matheus Schmitz
'''

# Imports
import sys
import time
import random
import numpy as np
import math
from sklearn import cluster


def point_cluster_mahalanobis_distance(datapoint_idx, cluster_stats):
    
    stdev = cluster_stats['stdev']
    centroid = cluster_stats['centroid']
    mahalanobis_distance = 0
    for dim in range(dataset_dimensionality):
        mahalanobis_distance += ((dataset_dict[datapoint_idx]['features'][dim] - centroid[dim]) / stdev[dim])**2
    mahalanobis_distance = np.sqrt(mahalanobis_distance)

    return mahalanobis_distance


def intercluster_mahalanobis_distance(left_cluster_stats, right_cluster_stats): # takes the dict value
    
    stdev1, centroid1 = left_cluster_stats['stdev'], left_cluster_stats['centroid']
    stdev2, centroid2 = right_cluster_stats['stdev'], right_cluster_stats['centroid']

    # Calculate the mahalanobis distance between the clusters
    left_cluster_dist, right_cluster_dist = 0, 0
    for dim in range(dataset_dimensionality):
        left_cluster_dist += ((centroid1[dim] - centroid2[dim]) / stdev2[dim])**2
        right_cluster_dist += ((centroid2[dim] - centroid1[dim]) / stdev1[dim])**2
    left_cluster_dist = np.sqrt(left_cluster_dist)
    right_cluster_dist = np.sqrt(right_cluster_dist)

    return min(left_cluster_dist, right_cluster_dist)


def update_DS(datapoint_idx, cluster_idx):
    DISCARD_SET_stats[cluster_idx]['ids_in_cluster'].append(datapoint_idx)
    DISCARD_SET_stats[cluster_idx]['N'] += 1
    for dim in range(dataset_dimensionality):
        DISCARD_SET_stats[cluster_idx]['sum'][dim] += dataset_dict[datapoint_idx]['features'][dim]
        DISCARD_SET_stats[cluster_idx]['sumsq'][dim] += dataset_dict[datapoint_idx]['features'][dim]**2
    DISCARD_SET_stats[cluster_idx]['stdev'] = np.sqrt(
                                              (DISCARD_SET_stats[cluster_idx]['sumsq'] / DISCARD_SET_stats[cluster_idx]['N']) - 
                                              (np.square(DISCARD_SET_stats[cluster_idx]['sum']) / DISCARD_SET_stats[cluster_idx]['N']**2)
                                         )
    DISCARD_SET_stats[cluster_idx]['centroid'] = DISCARD_SET_stats[cluster_idx]['sum'] / DISCARD_SET_stats[cluster_idx]['N']


def update_CS(datapoint_idx, cluster_idx):
    COMPRESSION_SET_stats[cluster_idx]['ids_in_cluster'].append(datapoint_idx)
    COMPRESSION_SET_stats[cluster_idx]['N'] += 1
    for dim in range(dataset_dimensionality):
        COMPRESSION_SET_stats[cluster_idx]['sum'][dim] += dataset_dict[datapoint_idx]['features'][dim]
        COMPRESSION_SET_stats[cluster_idx]['sumsq'][dim] += dataset_dict[datapoint_idx]['features'][dim]**2   
    COMPRESSION_SET_stats[cluster_idx]['stdev'] = np.sqrt(
                                              (COMPRESSION_SET_stats[cluster_idx]['sumsq'] / COMPRESSION_SET_stats[cluster_idx]['N']) - 
                                              (np.square(COMPRESSION_SET_stats[cluster_idx]['sum']) / COMPRESSION_SET_stats[cluster_idx]['N']**2)
                                         ) 
    COMPRESSION_SET_stats[cluster_idx]['centroid'] = COMPRESSION_SET_stats[cluster_idx]['sum'] / COMPRESSION_SET_stats[cluster_idx]['N']


def merge_CS_CS(key1, key2):
    COMPRESSION_SET_stats[key1]['ids_in_cluster'].extend(COMPRESSION_SET_stats[key2]['ids_in_cluster'])
    COMPRESSION_SET_stats[key1]['N'] += COMPRESSION_SET_stats[key2]['N']
    for dim in range(dataset_dimensionality):
        COMPRESSION_SET_stats[key1]['sum'][dim] += COMPRESSION_SET_stats[key2]['sum'][dim]
        COMPRESSION_SET_stats[key1]['sumsq'][dim] += COMPRESSION_SET_stats[key2]['sumsq'][dim]
    COMPRESSION_SET_stats[key1]['stdev'] = np.sqrt(
                                              (COMPRESSION_SET_stats[key1]['sumsq'] / COMPRESSION_SET_stats[key1]['N']) - 
                                              ((np.square(COMPRESSION_SET_stats[key1]['sum']) / COMPRESSION_SET_stats[key1]['N']**2))
                                         )
    COMPRESSION_SET_stats[key1]['centroid'] = COMPRESSION_SET_stats[key1]['sum'] / COMPRESSION_SET_stats[key1]['N']

    # Remove the merged cluster
    COMPRESSION_SET_stats.pop(key2, None)


def merge_CS_DS(keyCS, keyDS):
    DISCARD_SET_stats[keyDS]['ids_in_cluster'].extend(COMPRESSION_SET_stats[keyCS]['ids_in_cluster'])
    DISCARD_SET_stats[keyDS]['N'] += COMPRESSION_SET_stats[keyCS]['N']
    for dim in range(dataset_dimensionality):
        DISCARD_SET_stats[keyDS]['sum'][dim] += COMPRESSION_SET_stats[keyCS]['sum'][dim]
        DISCARD_SET_stats[keyDS]['sumsq'][dim] += COMPRESSION_SET_stats[keyCS]['sumsq'][dim]
    DISCARD_SET_stats[keyDS]['stdev'] = np.sqrt(
                                              (DISCARD_SET_stats[keyDS]['sumsq'] / DISCARD_SET_stats[keyDS]['N']) - 
                                              ((np.square(DISCARD_SET_stats[keyDS]['sum']) / DISCARD_SET_stats[keyDS]['N']**2))
                                         )
    DISCARD_SET_stats[keyDS]['centroid'] = DISCARD_SET_stats[keyDS]['sum'] / DISCARD_SET_stats[keyDS]['N']

    # Remove the merged cluster
    COMPRESSION_SET_stats.pop(keyCS, None)


if __name__ == "__main__":

    start_time = time.time()

    # Read user's inputs
    input_file = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file = sys.argv[3]

    # Read the input file (without using spark!)
    with open(input_file, 'r') as f_in:
        dataset = np.array(f_in.readlines())

    # From the file figure out the sample size for each of the 5 iterations
    sample_size = math.ceil(len(dataset)*0.2)

    # Conver the dataset to a dict for easily accessing the relevant information from a sample's id
    dataset_dict = {}
    for row in dataset:
        row_data = row.replace("\n","").split(",")
        dataset_dict[int(row_data[0])] = {'datapoint_id': int(row_data[0]), 
                                          'true_cluster': int(row_data[1]), 
                                          'features': [float(feat) for feat in row_data[2:]]}

    # List all sample_ids which have not yet been clustered
    unused_ids = set(dataset_dict.keys())

    # Get the dataset's dimensionality 
    dataset_dimensionality = len(dataset_dict[0]['features'])

    # Calculate the Mahalanobis threshold to be used
    mahalanobis_threshold = 2 * np.sqrt(dataset_dimensionality)

    # Initialize the objects for the different sets
    DS_CLUSTERS = dict()
    DISCARD_SET_stats = dict()
    COMPRESSION_SET_stats = dict()
    RETAINED_SET = list()

    # Sample the first round of data to initialize the algorithm
    round_1_samples = random.sample(unused_ids, sample_size)
    for sample_id in round_1_samples:
        unused_ids.remove(sample_id) 

    # Create the K-Means training data from the round_1_samples
    init_sample_features = []
    for sample_id in round_1_samples:
        init_sample_features.append(dataset_dict[sample_id]['features'])
    X_train = np.array(init_sample_features)

    # Run K-Means (e.g., from sklearn) with a large K (e.g., 5 times of the number of the input clusters)
    kmeans = cluster.KMeans(n_clusters=10*n_cluster)
    predicted_clusters = kmeans.fit_predict(X_train)

    # Assign points to clusters
    init_clusters = dict()
    for cluster_id, sampled_id in zip(predicted_clusters, round_1_samples):
        # If there is no key for the current cluster id, create one 
        if init_clusters.get(cluster_id) == None:
            init_clusters[cluster_id] = []
        init_clusters[cluster_id].append(sampled_id)

    # Move lone points to the RETAINED_SET
    for cluster_id, clustered_points in init_clusters.items():
        if len(clustered_points) == 1:
            RETAINED_SET.append(clustered_points[0])
            round_1_samples.remove(clustered_points[0])

    # Cluster the round_1_samples after having moved outliers to the RS
    train_sample_features = []
    for sample_id in round_1_samples:
        train_sample_features.append(dataset_dict[sample_id]['features'])
    X_train = np.array(train_sample_features)
    kmeans = cluster.KMeans(n_clusters=n_cluster)
    predicted_clusters = kmeans.fit_predict(X_train)

    # Assign points to DS clusters
    for cluster_id, sampled_id in zip(predicted_clusters, round_1_samples):
        # If there is no key for the current cluster id, create one 
        if DS_CLUSTERS.get(cluster_id) == None:
            DS_CLUSTERS[cluster_id] = []
        DS_CLUSTERS[cluster_id].append(sampled_id)

    # Calculate the cluster statistics for the clusters generated from the initialization round
    for key, value in DS_CLUSTERS.items():
        DISCARD_SET_stats[key] = {'ids_in_cluster': []}
        features_matrix = []
        for datapoint in value:
            DISCARD_SET_stats[key]['ids_in_cluster'].append(datapoint)
            features_matrix.append(dataset_dict[datapoint]['features'])
        features_matrix = np.array(features_matrix)
        DISCARD_SET_stats[key]['N'] = len(DISCARD_SET_stats[key]['ids_in_cluster'])
        DISCARD_SET_stats[key]['sum'] = features_matrix.sum(axis=0)
        DISCARD_SET_stats[key]['sumsq'] = np.sum(features_matrix**2, axis=0)
        DISCARD_SET_stats[key]['stdev'] = np.sqrt(
                                                  (DISCARD_SET_stats[key]['sumsq']/DISCARD_SET_stats[key]['N']) - 
                                                  (np.square(DISCARD_SET_stats[key]['sum']) / DISCARD_SET_stats[key]['N']**2)
                                             )
        DISCARD_SET_stats[key]['centroid'] = DISCARD_SET_stats[key]['sum']/DISCARD_SET_stats[key]['N']

    # Run KMeans on the RETAINED_SET and generate COMPRESSION_SET clusters
    RS_sample_features = []
    for sample_id in RETAINED_SET:
        RS_sample_features.append(dataset_dict[sample_id]['features'])
    X_train = np.array(RS_sample_features)
    # If there were any samples in the RS, cluster them
    RS_clusters = {}
    if X_train.shape[0] > 0:
        kmeans = cluster.KMeans(n_clusters=min(7*n_cluster, int(1 + (X_train.shape[0]/2))))
        predicted_clusters = kmeans.fit_predict(X_train)
        for cluster_id, sampled_id in zip(predicted_clusters, RETAINED_SET):
            # If there is no key for the current cluster id, create one
            if RS_clusters.get(cluster_id) == None:
                RS_clusters[cluster_id] = list()
            # Then append the RS sample to the its newly found CS cluster
            RS_clusters[cluster_id].append(sampled_id)


    # Calculate the cluster statistics for the clusters generated from the initialization round
    for key, value in RS_clusters.items():
        
        # Only clusters with more than one sample in them go to the CS, those with only 1 sample remain RS 
        if len(value) > 1:
        
            COMPRESSION_SET_stats[key] = {'ids_in_cluster': []}
            features_matrix = []
            for datapoint in value:
                COMPRESSION_SET_stats[key]['ids_in_cluster'].append(datapoint)
                features_matrix.append(dataset_dict[datapoint]['features'])
            features_matrix = np.array(features_matrix)
            COMPRESSION_SET_stats[key]['N'] = len(COMPRESSION_SET_stats[key]['ids_in_cluster'])
            COMPRESSION_SET_stats[key]['sum'] = features_matrix.sum(axis=0)
            COMPRESSION_SET_stats[key]['sumsq'] = np.sum(features_matrix**2, axis=0)
            COMPRESSION_SET_stats[key]['stdev'] = np.sqrt(
                                                      (COMPRESSION_SET_stats[key]['sumsq']/COMPRESSION_SET_stats[key]['N']) - 
                                                      (np.square(COMPRESSION_SET_stats[key]['sum']) / COMPRESSION_SET_stats[key]['N']**2)                                                     
                                                 )
            COMPRESSION_SET_stats[key]['centroid'] = COMPRESSION_SET_stats[key]['sum']/COMPRESSION_SET_stats[key]['N']
            
            # Clean the samples from the retained set which have been summarized
            for datapoint in value:
                RETAINED_SET.remove(datapoint)

    # Objects to tally values to be outputted
    num_DS_points = 0 
    num_CS_clusters = 0
    num_CS_points = 0
    num_RS_points = 0

    # Tally the values for outputting
    for key, value in DISCARD_SET_stats.items():
        num_DS_points += value['N']
    for key, value in COMPRESSION_SET_stats.items():
        num_CS_clusters += 1
        num_CS_points += value['N']
    num_RS_points = len(RETAINED_SET)

    # Write the results for the initialization round (round 1)
    f_out = open(output_file, "w")
    f_out.write("The intermediate results:")
    f_out.write("\n" + "Round 1: " + str(num_DS_points) + "," + str(num_CS_clusters) + "," + str(num_CS_points) + "," + str(num_RS_points))


    ### Initialization is finished, run the regular iterations ###
    for curr_round in range(2,6):
        
        # Load the samples for the iteration about the begin
        if curr_round < 5:
            iteration_samples = random.sample(unused_ids, sample_size)
            for sample_id in iteration_samples:
                unused_ids.remove(sample_id) 
        # The last iteration might have a slightly different number of samples
        else:
            iteration_samples = list(unused_ids.copy())
            for sample_id in iteration_samples:
                unused_ids.remove(sample_id) 
        
        
        ##################################
        ### Assign Samples to DS/CS/RS ###
        ##################################
        
        # Iterate over each sample among those drawn for this iterations
        for sample_id in iteration_samples:
                        
            # Default to assuming points are in the RETAINED_SET
            assigned_cluster = -1    

            # Track the lowest distance between the sample and all clusters
            lowest_dist = mahalanobis_threshold

            # Find the DISCARD_SET cluster closest to the current sample      
            for cluster_id, cluster_stats in DISCARD_SET_stats.items():

                mahalanobis_distance = point_cluster_mahalanobis_distance(sample_id, cluster_stats)

                # If the distance is under the mahalanobis_threshold and also the lowest distance yet found, update the point's cluster
                if mahalanobis_distance < lowest_dist:
                    assigned_cluster = cluster_id
                    lowest_dist = mahalanobis_distance

            # Update the statistics of the cluster to which the point is assigned 
            if assigned_cluster != -1:
                update_DS(sample_id, assigned_cluster)
                
            # If the sample could not be assigned to any cluster in the DS, try assigning it to the CS
            else:
                
                # Find the COMPRESSION_SET cluster closest to the current sample
                for cluster_id, cluster_stats in COMPRESSION_SET_stats.items():
                    
                    mahalanobis_distance = point_cluster_mahalanobis_distance(sample_id, cluster_stats)
                    
                    # If the distance is under the mahalanobis_threshold and also the lowest distance yet found, update the point's cluster
                    if mahalanobis_distance < lowest_dist:
                        assigned_cluster = cluster_id
                        lowest_dist = mahalanobis_distance

                # Update the statistics of the cluster to which the point is assigned 
                if assigned_cluster != -1:
                    update_CS(sample_id, assigned_cluster)

                # If the BRF also failed to assing the sample to a CS cluster, send it to the RS
                else:
                    RETAINED_SET.append(sample_id)
        

        ######################################
        ### Create New CSs from RS Samples ###
        ######################################

        # Run KMeans on the RETAINED_SET and generate COMPRESSION_SET clusters
        RS_sample_features = []
        for sample_id in RETAINED_SET:
            RS_sample_features.append(dataset_dict[sample_id]['features'])
        X_train = np.array(RS_sample_features)
        RS_clusters = {}
        if X_train.shape[0] > 0:
            kmeans = cluster.KMeans(n_clusters=min(7*n_cluster, int(1 + (X_train.shape[0]/2))))
            predicted_clusters = kmeans.fit_predict(X_train)
            for cluster_id, sampled_id in zip(predicted_clusters, RETAINED_SET):
                # If there is no key for the current cluster id, create one
                if RS_clusters.get(cluster_id) == None:
                    RS_clusters[cluster_id] = list()
                # Then append the RS sample to the its newly found CS cluster
                RS_clusters[cluster_id].append(sampled_id)


        # Calculate the cluster statistics for the clusters generated from the initialization round
        for key, value in RS_clusters.items():

            # Only clusters with more than one sample in them go to the CS, those with only 1 sample remain RS 
            if len(value) > 1:
                
                # Find the find the next cluster index to use for the COMPRESSION_SET_stats
                try:
                    CS_stats_next_key = max(COMPRESSION_SET_stats.keys()) + 1
                # If there are no keys from which to get the max value, then start at key 0
                except:
                    CS_stats_next_key = 0

                COMPRESSION_SET_stats[CS_stats_next_key] = {'ids_in_cluster': []}
                features_matrix = []
                for datapoint in value:
                    COMPRESSION_SET_stats[CS_stats_next_key]['ids_in_cluster'].append(datapoint)
                    features_matrix.append(dataset_dict[datapoint]['features'])
                features_matrix = np.array(features_matrix)
                COMPRESSION_SET_stats[CS_stats_next_key]['N'] = len(COMPRESSION_SET_stats[CS_stats_next_key]['ids_in_cluster'])
                COMPRESSION_SET_stats[CS_stats_next_key]['sum'] = features_matrix.sum(axis=0)
                COMPRESSION_SET_stats[CS_stats_next_key]['sumsq'] = np.sum(features_matrix**2, axis=0)
                COMPRESSION_SET_stats[CS_stats_next_key]['stdev'] = np.sqrt(
                                                          (COMPRESSION_SET_stats[CS_stats_next_key]['sumsq'] / COMPRESSION_SET_stats[CS_stats_next_key]['N']) -
                                                          (np.square(COMPRESSION_SET_stats[CS_stats_next_key]['sum']) / COMPRESSION_SET_stats[CS_stats_next_key]['N']**2) 
                                                     )
                COMPRESSION_SET_stats[CS_stats_next_key]['centroid'] = COMPRESSION_SET_stats[CS_stats_next_key]['sum'] / COMPRESSION_SET_stats[CS_stats_next_key]['N']

                # Clean the samples from the retained set which have been summarized
                for datapoint in value:
                    RETAINED_SET.remove(datapoint)
                           

        #################################################
        ### Merge CSs Below the Mahalanobis Threshold ###
        #################################################
        
        close_CSs = dict()
        for key1, value1 in COMPRESSION_SET_stats.items():
            
            # Default to assuming there is no other CS cluster close by
            assigned_cluster = None

            # Track the lowest distance between the sample and all clusters
            lowest_dist = mahalanobis_threshold
            
            # Compare each CS cluster to all other CS clusters
            for key2, value2 in COMPRESSION_SET_stats.items():
                
                # Do not compare the a cluster to itself
                if key1 == key2:
                    continue
                    
                intercluster_dist = intercluster_mahalanobis_distance(value1, value2)
                
                # If the intercluster distance is below the threshold, make the pair a candidate for merging
                if intercluster_dist < lowest_dist:
                    assigned_cluster = key2
                    lowest_dist = intercluster_dist
                    
            # Once all pairwise comparisons were done for a given cluster (key1) store the results
            close_CSs[key1] = assigned_cluster
            
        # Once all closest CS clusters were found, merge them and update the COMPRESSION_SET_stats
        for CS_cluster1, CS_cluster2 in close_CSs.items():
            if CS_cluster1 in COMPRESSION_SET_stats and CS_cluster2 in COMPRESSION_SET_stats and CS_cluster1 != CS_cluster2:
                merge_CS_CS(CS_cluster1, CS_cluster2)
                

        ######################################################
        ### At the Final Iteration Merge CSs to Nearby DSs ###
        ######################################################
        
        # Check if it is the last iteration
        if curr_round == 5:
            
            close_DSs = dict()
            for CS_key, CS_value in COMPRESSION_SET_stats.items():
                
                # Default to assuming there is no DS cluster close by
                assigned_cluster = None

                # Track the lowest distance between the sample and all clusters
                lowest_dist = mahalanobis_threshold

                for DS_key, DS_value in DISCARD_SET_stats.items():
                    
                    intercluster_dist = intercluster_mahalanobis_distance(CS_value, DS_value)

                    # If the intercluster distance is below the threshold, make the pair a candidate for merging
                    if intercluster_dist < lowest_dist:
                        assigned_cluster = DS_key
                        lowest_dist = intercluster_dist
                    
                # Once all pairwise comparisons were done for a given cluster (CS_key) store the results
                close_DSs[CS_key] = assigned_cluster
            
            # Once all closest CS clusters were found, merge them and update the COMPRESSION_SET_stats
            for CS_cluster, DS_cluster in close_DSs.items():
                if CS_cluster in COMPRESSION_SET_stats and DS_cluster in DISCARD_SET_stats:
                    merge_CS_DS(CS_cluster, DS_cluster)
                    

        ###############################
        ### Output Round Statistics ###
        ###############################

        # Objects to tally values to be outputted
        num_DS_points = 0 
        num_CS_clusters = 0
        num_CS_points = 0
        num_RS_points = 0
        
        # Tally the values for outputting
        for key, value in DISCARD_SET_stats.items():
            num_DS_points += value['N']
        for key, value in COMPRESSION_SET_stats.items():
            num_CS_clusters += 1
            num_CS_points += value['N']
        num_RS_points = len(RETAINED_SET)
        
        # Add the current round's results to the output file
        f_out.write("\n" + "Round " + str(curr_round) + ": " + str(num_DS_points) + "," + str(num_CS_clusters) + "," + str(num_CS_points) + "," + str(num_RS_points))


    # Go over all sets (DS, CS, RS) and extract each datapoint's final cluster
    sample_id_to_cluster = dict()
    for key, value in DISCARD_SET_stats.items():
        for datapoint in value['ids_in_cluster']:
            sample_id_to_cluster[datapoint] = key
    for key, value in COMPRESSION_SET_stats.items():
        for datapoint in value['ids_in_cluster']:
            sample_id_to_cluster[datapoint] = -1
    for datapoint in RETAINED_SET:
        sample_id_to_cluster[datapoint] = -1

    # Write the clustering results
    f_out.write("\n" + "\n" + "The clustering results:")
    for key, value in sorted(sample_id_to_cluster.items()):
        f_out.write("\n" + str(key) + "," + str(value))

    print(f'Duration: {time.time() - start_time} seconds.')