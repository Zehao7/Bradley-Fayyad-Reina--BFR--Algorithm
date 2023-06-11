import numpy as np
from sklearn.cluster import KMeans
import math
import time
import sys
np.random.seed(666)


# spark-submit task.py <input_file> <n_cluster> <output_file>
# spark-submit task.py "/Users/leoli/Desktop/hw6_clustering.txt" 10 "/Users/leoli/Desktop/hw6_output.txt"



if __name__ == "__main__":


    start_time = time.time()


    # Step 1. Load 20% of the data randomly.
    # input_file = "/Users/leoli/Desktop/hw6_clustering.txt"
    # n_cluster = 10
    # output_file = "/Users/leoli/Desktop/hw6_output.txt"
    input_file = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file = sys.argv[3]

    cluster_threshold = 5
    large_k = 5 * n_cluster


    # df = np.loadtxt(input_file, delimiter=',')[:, 2:]
    df = np.loadtxt(input_file, delimiter=',')
    np.random.shuffle(df)
    fraction = 0.2

    def splitData(df, fraction):
        len_df = len(df)
        num_data_portions = 1 / fraction
        newdata = dict()
        for i in range(int(num_data_portions)):
            newdata[i] = df[int(i*fraction*len_df):int((i+1)*fraction*len_df)]
        return newdata

    df = splitData(df, fraction)


    def get_centroid(cluster):
        n = cluster[0]
        SUM = cluster[1]
        centroid = SUM / n
        return centroid
    # print(get_centroid(DS[0]))


    def get_variance(cluster):
        n = cluster[0]
        SUM = cluster[1]
        SUMSQ = cluster[2]
        variance = [(sumsq_i / n) - (sum_i / n) ** 2 for sumsq_i, sum_i in zip(SUMSQ, SUM)]
        return variance
        

    def mahalanobis_distance(x, cluster):
        centroid = get_centroid(cluster)
        variance = get_variance(cluster)
        # print("centroid: ", centroid)
        # print("variance: ", variance)
        # md = np.sqrt(np.sum((x - centroid) ** 2 / variance))
        md = sum([((x_i - c_i) / math.sqrt(sigma_i)) ** 2 for x_i, c_i, sigma_i in zip(x, centroid, variance)])
        return math.sqrt(md)



    intermediate_result = {}
    cluster_result = {}
    df2 = np.loadtxt(input_file, delimiter=',')
    for i in range(len(df2)):
        cluster_result[int(df2[i][0])] = int(df2[i][1])
    # print(cluster_result[0:20])



    for i in range(len(df)):
    # for i in range(2):
        num_round = i + 1
        if i == 0:
            # Step 2. Run K-Means (e.g., from sklearn) with a large K (e.g., 5 times of the number of the input clusters) on the data in memory using the Euclidean distance as the similarity measurement.
            # print("-------------------- Step 2 --------------------")
            first_chunk = df[0]
            # kmeans = KMeans(n_clusters=large_k, random_state=42).fit(first_chunk[:, 1:])
            # first_chunk_kmeans = KMeans(n_clusters=large_k, n_init=10).fit(first_chunk)
            first_chunk_kmeans = KMeans(n_clusters=large_k, n_init=10).fit(first_chunk[:, 2:])


            # Step 3. In the K-Means result from Step 2, move all the clusters that contain only one point to RS (outliers).
            # print("-------------------- Step 3 --------------------")
            init_RS = []
            rest_data = []
            for i in range(first_chunk_kmeans.n_clusters):
                cluster_i = first_chunk[first_chunk_kmeans.labels_ == i]
                n = len(cluster_i)
                if n < cluster_threshold:
                    init_RS.extend(cluster_i)
                else:
                    rest_data.extend(cluster_i)


            # print("len of init_RS: ", len(init_RS))
            # print(init_RS)




            # Step 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters.
            # print("-------------------- Step 4 --------------------")
            rest_data = np.array(rest_data)
            # rest_data_kmeans = KMeans(n_clusters=n_cluster, n_init=10).fit(rest_data[:, 1:])
            # rest_data_kmeans = KMeans(n_clusters=n_cluster, n_init=10).fit(rest_data)
            rest_data_kmeans = KMeans(n_clusters=n_cluster, n_init=10).fit(rest_data[:, 2:])



            # Step 5. Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and generate statistics).
            # print("-------------------- Step 5 --------------------")
            DS = {}
            for i in range(rest_data_kmeans.n_clusters):
                cluster_i = rest_data[rest_data_kmeans.labels_ == i]
                n = len(cluster_i)
                SUM = np.sum(cluster_i[:, 2:], axis=0)
                # SUM = np.sum(cluster_i, axis=0)
                SUMSQ = np.sum(cluster_i[:, 2:]**2, axis=0)
                # SUMSQ = np.sum(cluster_i**2, axis=0)
                DS[i] = [n, SUM, SUMSQ]
                # DS[i] = [n, SUM, SUMSQ, cluster_i]
                # cluster_result.append([int(cluster_i[j][0]), i] for j in range(len(cluster_i)))
                for j in range(len(cluster_i)):
                    cluster_result[int(cluster_i[j][0])] = i


            # DS: {cluster_id: [n, SUM, SUMSQ]}
            # print("len of DS: ", len(DS))
            # print(DS[0])



            # The initialization of DS has finished, so far, you have K numbers of DS clusters (from Step 5) and some numbers of RS (from Step 3).
            # print("variance: ", get_variance(DS[0]))


            # Step 6. Run K-Means on the points in the RS with a large K (e.g., 5 times of the number of the input clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point).
            # print("-------------------- Step 6 --------------------")
            min_cluster = min(n_cluster, len(init_RS))
            init_RS = np.array(init_RS)
            # init_RS_kmeans = KMeans(n_clusters=n_cluster, n_init=10).fit(init_RS)
            init_RS_kmeans = KMeans(n_clusters=min_cluster, n_init=10).fit(init_RS[:, 2:])
            CS = dict()
            RS = list()
            idx = 0
            for i in range(init_RS_kmeans.n_clusters):
                cluster_i = init_RS[init_RS_kmeans.labels_ == i]
                n = len(cluster_i)
                if n <= 1:
                    RS.extend(cluster_i)
                else:
                    SUM = np.sum(cluster_i[:, 2:], axis=0)
                    # SUM = np.sum(cluster_i, axis=0)
                    SUMSQ = np.sum(cluster_i[:, 2:]**2, axis=0)
                    # SUMSQ = np.sum(cluster_i**2, axis=0)
                    CS[idx] = [n, SUM, SUMSQ, cluster_i]
                    idx += 1

            # print("len of new RS: ", len(RS))
            # print(RS)

            # CS: {cluster_id: [n, SUM, SUMSQ, cluster_i]}
            # print("len of CS: ", len(CS))
            # print(CS)

            # print("==============================================")
            # print("\n")


            # Intermediate Result
            # print("-------------------- Intermediate Result --------------------")
            num_round_string = "Round "+ str(num_round) + ":"

            # the number of the discard points
            num_discard_points = sum(DS[i][0] for i in range(len(DS)))

            # the number of the clusters in the CS
            num_cs_clusters = len(CS)

            # the number of the compression points
            num_compression_points = sum(CS[i][0] for i in range(len(CS)))

            # the number of the points in the retained set
            num_retained_points = len(RS)

            # print("num_round, num_discard_points, num_cs_clusters, num_compression_points, num_retained_points")
            # print(num_round_string, num_discard_points, num_cs_clusters, num_compression_points, num_retained_points)

            intermediate_result[num_round_string] = [num_discard_points, num_cs_clusters, num_compression_points, num_retained_points]



        else:
            # Step 7. Load another 20% of the data randomly.
            # print("-------------------- Step 7: new round --------------------")
            next_chunk = df[i]

            # d1 =  mahalanobis_distance(get_centroid(DS[0]), DS[1])
            # d2 =  mahalanobis_distance(get_centroid(DS[1]), DS[0])
            # print("d1: ", d1)
            # print("d2: ", d2)

            # Step 8. For the new points, compare them to each of the DS using the Mahalanobis Distance and assign them to the nearest DS clusters if the distance is < 2*sqrt(d).
            # print("-------------------- Step 8: add new points to DS --------------------")
            dimension = len(next_chunk[0]) - 2
            threshold = 2 * np.sqrt(dimension)

            def add_to_DS(points_list, ds_dict):
                temp_ds_dict = ds_dict.copy()
                n = len(temp_ds_dict)
                m = len(points_list)
                ds_left = []
                count = 0
                for i in range(m):
                    # distances = [mahalanobis_distance(points_list[i], temp_ds_dict[j]) for j in range(n)]
                    distances = [mahalanobis_distance(points_list[i][2:], temp_ds_dict[j]) for j in range(n)]
                    min_md = min(distances)
                    min_index = distances.index(min_md)
                    if min_md > threshold:
                        ds_left.append(points_list[i])
                    else:
                        ds_dict[min_index][0] += 1
                        ds_dict[min_index][1] += points_list[i][2:]
                        ds_dict[min_index][2] += points_list[i][2:]**2
                        # ds_dict[min_index][3] = np.vstack((ds_dict[min_index][3], points_list[i]))
                        cluster_result[int(points_list[i][0])] = min_index
                        count += 1

                # print("num add to DS count: ", count)
                return ds_left, ds_dict
            
            ds_left, DS = add_to_DS(next_chunk, DS)
            
            # print("len of ds_left: ", len(ds_left))
            # print("len of DS: ", len(DS))





            # Step 9. For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and assign the points to the nearest CS clusters if the distance is < 2*sqrt(d).
            # print("-------------------- Step 9: add ds_left to CS --------------------")
            def add_to_CS(points_list, cs_dict):
                if len(points_list) == 0 or len(cs_dict) == 0:
                    return [], cs_dict
                
                temp_cs_dict = cs_dict.copy()
                n = len(temp_cs_dict)
                m = len(points_list)
                cs_left = []
                count = 0
                for i in range(m):
                    # distances = [mahalanobis_distance(points_list[i], temp_cs_dict[j]) for j in range(n)]
                    distances = [mahalanobis_distance(points_list[i][2:], temp_cs_dict[j]) for j in range(n)]
                    min_md = min(distances)
                    min_index = distances.index(min_md)
                    if min_md > threshold:
                        cs_left.append(points_list[i])
                    else:
                        cs_dict[min_index][0] += 1
                        cs_dict[min_index][1] += points_list[i][2:]
                        cs_dict[min_index][2] += points_list[i][2:]**2
                        cs_dict[min_index][3] = np.vstack((cs_dict[min_index][3], points_list[i]))
                        count += 1
                    
                # print("num add to CS count: ", count)
                return cs_left, cs_dict
            
            cs_left, CS = add_to_CS(ds_left, CS)

            # print("len of cs_left: ", len(cs_left))
            # print(cs_left[0])
            # print("len of CS: ", len(CS))
            # print(CS)





            # Step 10. For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.
            # print("-------------------- Step 10: add cs_left to RS --------------------")
            RS.extend(cs_left)
            # print("len of RS: ", len(RS))
            # print(RS[0])



            # Step 11. Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters) to generate CS (clusters with more than one points) and RS (clusters with only one point).
            # print("-------------------- Step 11: Run K-Means on RS to create CS and RS --------------------")
            min_cluster = min(n_cluster, len(RS))
            temp_RS = np.array(RS)

            # temp_RS_kmeans = KMeans(n_clusters=min_cluster, n_init=10).fit(temp_RS)
            temp_RS_kmeans = KMeans(n_clusters=min_cluster, n_init=10).fit(temp_RS[:, 2:])
            new_CS = dict()
            RS = list()
            idx = 0
            for i in range(temp_RS_kmeans.n_clusters):
                cluster_i = temp_RS[temp_RS_kmeans.labels_ == i]
                n = len(cluster_i)
                if n <= 1:
                    RS.extend(cluster_i)
                else:
                    # SUM = np.sum(cluster_i, axis=0)
                    SUM = np.sum(cluster_i[:, 2:], axis=0)
                    # SUMSQ = np.sum(cluster_i**2, axis=0)
                    SUMSQ = np.sum(cluster_i[:, 2:]**2, axis=0)
                    new_CS[idx] = [n, SUM, SUMSQ, cluster_i]
                    idx += 1


            # print("len of RS after create new_CS: ", len(RS))
            # print(RS)
            # print("len of new_CS: ", len(new_CS))
            # print(new_CS)
                
            
            # d1 = mahalanobis_distance(get_centroid(CS[0]), CS[1])
            # d2 = mahalanobis_distance(get_centroid(CS[1]), CS[0])
            # print("d1: ", d1)
            # print("d2: ", d2)
            # new_cs_dict = dict()
            # new_cs_dict[0] = [CS[0][0]+CS[1][0], CS[0][1]+CS[1][1], CS[0][2]+CS[1][2], np.vstack((CS[0][3], CS[1][3]))]
            # new_cs_dict[1] = [CS[1][0]+CS[0][0], CS[1][1]+CS[0][1], CS[1][2]+CS[0][2], np.vstack((CS[1][3], CS[0][3]))]
            # print("new_cs_dict: ", new_cs_dict)
            



            # Step 12. Merge CS clusters that have a Mahalanobis Distance < 2*sqrt(d).
            # print("-------------------- Step 12 Merge new and old CS clusters --------------------")
            # print(CS)

            def merge_CS_clusters(old_cs, new_cs):
                n = len(old_cs)
                m = len(new_cs)
                merged = set()
                copy_old_cs = old_cs.copy()
                for i in range(n):
                    for j in range(i+1, m):
                        d1 = mahalanobis_distance(get_centroid(old_cs[i]), new_cs[j])
                        d2 = mahalanobis_distance(get_centroid(new_cs[j]), old_cs[i])
                        # pairs = (i, j) if d1 > d2 else (j, i)
                        # md = max(d1, d2)
                        md = np.average([d1, d2])
                        if md < threshold and j not in merged:
                            merged.add(j)
                            copy_old_cs[i] = [copy_old_cs[i][0] + new_cs[j][0], copy_old_cs[i][1] + new_cs[j][1], copy_old_cs[i][2] + new_cs[j][2], np.vstack((copy_old_cs[i][3], new_cs[j][3]))]
                idx = n
                for j in range(m):
                    if j not in merged:
                        copy_old_cs[idx] = new_cs[j]
                        idx += 1
                return copy_old_cs
            
            CS = merge_CS_clusters(CS, new_CS)

            # print("len of CS after merging: ", len(CS))
            # print(CS)



            # If this is the last run (after the last chunk of data), merge CS clusters with DS clusters that have a Mahalanobis Distance < 2*sqrt(d).
            if num_round == len(df):
                # print("-------------------- Last Run --------------------")
                def merge_CS_DS(cs_dict, ds_dict):
                    n = len(cs_dict)
                    m = len(ds_dict)
                    final_cs = dict()
                    copy_ds_dict = ds_dict.copy()
                    mergedCS = set()
                    idx = 0
                    for i in range(m):
                        for j in range(n):
                            d1 = mahalanobis_distance(get_centroid(ds_dict[i]), cs_dict[j])
                            d2 = mahalanobis_distance(get_centroid(cs_dict[j]), ds_dict[i])
                            md = np.average([d1, d2])
                            if md < threshold and j not in mergedCS:
                                mergedCS.add(j)
                                copy_ds_dict[i] = [copy_ds_dict[i][0] + cs_dict[j][0], copy_ds_dict[i][1] + cs_dict[j][1], copy_ds_dict[i][2] + cs_dict[j][2]]
                                # copy_ds_dict[i] = [copy_ds_dict[i][0] + cs_dict[j][0], copy_ds_dict[i][1] + cs_dict[j][1], copy_ds_dict[i][2] + cs_dict[j][2], np.vstack((copy_ds_dict[i][3], cs_dict[j][3]))]
                                for cs_dict_i in range(len(cs_dict[j][3])):
                                    cluster_result[cs_dict[j][3][cs_dict_i][0]] = i

                    
                    for i in range(n):
                        if i not in mergedCS:
                            final_cs[idx] = cs_dict[i]
                            idx += 1

                    # print("merged CS clusters: ", mergedCS)
                    return final_cs, copy_ds_dict

                CS, DS = merge_CS_DS(CS, DS)





            # Intermediate Result
            # print("-------------------- Intermediate Result --------------------")
            num_round_string = "Round "+ str(num_round) + ":"

            # the number of the discard points
            num_discard_points = sum(DS[i][0] for i in range(len(DS)))

            # the number of the clusters in the CS
            num_cs_clusters = len(CS)

            # the number of the compression points
            num_compression_points = sum(CS[i][0] for i in range(len(CS)))

            # the number of the points in the retained set
            num_retained_points = len(RS)

            # print("num_round, num_discard_points, num_cs_clusters, num_compression_points, num_retained_points")
            # print(num_round_string, num_discard_points, num_cs_clusters, num_compression_points, num_retained_points)

            intermediate_result[num_round_string] = [num_discard_points, num_cs_clusters, num_compression_points, num_retained_points]


            # Repeat Steps 7 – 12.

            # print("==============================================")
            # print("\n")



    # for i in range(len(CS)):
    #     num, SUM, sum_of_squares, points = CS[i]
    #     for j in range(len(points)):
    #         idx, clusterid = points[j][:2]
    #         cluster_result[int(idx)] = -1

    # for i in range(len(RS)):
    #     idx, clusterid = RS[i][:2]
    #     cluster_result[int(idx)] = -1



    # Final Result
    print("-------------------- Final Result --------------------")
    with open(output_file, 'w') as f:
        f.write("The intermediate results:\n")
        for key, value in intermediate_result.items():
            f.write(str(key) + " ")
            f.write(",".join(str(val) for val in value))
            f.write("\n")

        f.write("\n")
        f.write("The clustering results:\n")
        for key, value in cluster_result.items():
            f.write(str(key) + "," + str(value))
            f.write("\n")
            
    print("time: {0:.5f}".format(time.time() - start_time))


    # # At each run, including the initialization step, you need to count and output the number of the discard points, the number of the clusters in the CS, the number of the compression points, and the number of the points in the retained set.
