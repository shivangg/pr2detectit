#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms, compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray, DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

 
# RGB to HSV convertor
def rgb_to_hsv(rgb_list):
    rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
    hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
    return hsv_normalized

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    ## Statistical filtering
    outlier_filter = cloud.make_statistical_outlier_filter()

    # Number of points to analyse in the neighbourhood
    outlier_filter.set_mean_k(70)

    # set threshold filter
    x = 0.00005

    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)

    # Finally call the filter function for magic
    cloud = outlier_filter.filter()

    # publish the results of Statistical outlier filtering
    statistical_pub.publish(pcl_to_ros(cloud))


    # Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.01
    # a cube of leaf size
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # publish the results of vowel downsampling
    vowel_pub.publish(pcl_to_ros(cloud_filtered))

    ## PassThrough Filter 

    # passthrough in z direction
    passthrough_z = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough_z.set_filter_field_name(filter_axis)
    axis_min = 0.608
    axis_max = 1.0
    passthrough_z.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough_z.filter()

    # passthrough in the x direction
    passthrough_x = cloud_filtered.make_passthrough_filter()
    filter_axis = 'x'
    passthrough_x.set_filter_field_name(filter_axis)
    axis_min = 0.37
    axis_max = 10.37
    passthrough_x.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough_x.filter()

    # publish the results of Statistical outlier filtering
    passthrough_pub.publish(pcl_to_ros(cloud_filtered))
 
    ## RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
 
    # set the model which has to be fit
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # max distance for the point to be considered fit
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    ## Extract inliers and outliers
   
    inliers, coefficients = seg.segment()
    extracted_outliers = cloud_filtered.extract(inliers, negative = True)
    extracted_inliers = cloud_filtered.extract(inliers, negative = False)

    ## Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()

    # create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()

    # set tolerance for distance threshold
    # as well as max. and min. cluster size(in points)
    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(1200)

    # search the kd tree for clusters
    ec.set_SearchMethod(tree)
    
    # extract indices for each of the discovered cluster
    cluster_indices = ec.Extract()

    ## Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))

    # list to store the different point cloud clusters list 
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])
                                        ])
 
    # create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
 
    # Convert PCL data to ROS messages
    pcl_to_ros_objects = pcl_to_ros(extracted_outliers)
    pcl_to_ros_table = pcl_to_ros(extracted_inliers)

    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
 
    # Publish ROS messages
    pcl_objects_pub.publish(pcl_to_ros_objects)
    pcl_table_pub.publish(pcl_to_ros_table)

    pcl_cluster_pub.publish(ros_cluster_cloud)

    # Classify the clusters!
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
            # Grab the points for the cluster from the extracted outliers (cloud_objects)
            pcl_cluster = extracted_outliers.extract(pts_list)
            ## Convert the cluster from pcl to ROS using helper function
            ros_cluster = pcl_to_ros(pcl_cluster)

            # Extract histogram features
            #Compute the color and surface normal histograms
            chist = compute_color_histograms(ros_cluster, using_hsv = True )
            normals = get_normals(ros_cluster)
            normal_hist = compute_normal_histograms(normals)

            feature = np.concatenate((chist, normal_hist))

            # Make the prediction, retrieve the label for the result
            # and add it to detected_objects_labels list
            prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
            label = encoder.inverse_transform(prediction)[0]
            detected_objects_labels.append(label)
 
            # Publish a label into RViz
            label_pos = list(white_cloud[pts_list[0]])
            label_pos[2] += .4
            object_markers_pub.publish(make_label(label,label_pos, index))

            # Add the detected object to the list of detected objects.
            do = DetectedObject()
            do.label = label
            do.cloud = ros_cluster
            detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
 
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(detected_objects):

    ## Initialize variables
    test_scene_num = Int32()
    arm_name = String()
    object_name = String()
    pick_pose = Pose()
    place_pose = Pose()

    test_scene_num.data = 2
    detected_object_list = []
    yaml_list = []
    for i in range(len(detected_objects)):
        # print(detected_objects[i].label)
        detected_object_list.append(detected_objects[i].label)

    yaml_dict  = {}
    filename = "output.yaml"
    object_group = ''
    labels = []
    centroids = []
    object_list = []
    
    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    
    # Run for all the objects in the list
    for i in range(len(object_list_param)):
        object_list.append( (object_list_param[i]['name'], object_list_param[i]['group']) )

    # Loop through the pick list
    for item, group in object_list:
        print("item, group in yamla",item, group, "detected", detected_object_list)
        if item in detected_object_list:
            labels = [item]
            object_group = group
            index_item = detected_object_list.index(item)
            
            ## Get the PointCloud for a given object and obtain it's centroid
            points_arr = ros_to_pcl( detected_objects[detected_object_list.index(item)].cloud).to_array()
            position_mean_list = np.mean(points_arr, axis = 0)[:3]
            centroids = [[np.asscalar(position_mean_list[0]), np.asscalar(position_mean_list[1]), np.asscalar(position_mean_list[2])]]
            # print("centroid", centroids)

            ## Create 'pick_pose' for the object   

            # Pose() for pick_pose 
            pick_pose.position.x = centroids[0][0]
            pick_pose.position.y = centroids[0][1]
            pick_pose.position.z = centroids[0][2]

            pick_pose.orientation.x = 0
            pick_pose.orientation.y = 0
            pick_pose.orientation.z = 0
            pick_pose.orientation.w = 1

            # Create 'place_pose' for the object

            # Parse the information of the dropboxes
            place_point_position = rospy.get_param('/dropbox')

            place_pose.orientation.x = 0
            place_pose.orientation.y = 0
            place_pose.orientation.z = 0
            place_pose.orientation.w = 1

            ## The name of the object to be picked
            object_name.data = labels[0]

            ## Assign the arm to be used for pick_place
            if object_group is 'red':
                arm_name.data = 'left'
                # to get the place coordinates of left boxes 
                place_pose.position.x = place_point_position[0]['position'][0]
                place_pose.position.y = place_point_position[0]['position'][1]
                place_pose.position.z = place_point_position[0]['position'][2]
            else:
                arm_name.data = 'right'
                # to get the place coordinates of right boxes 
                place_pose.position.x = place_point_position[1]['position'][0]
                place_pose.position.y = place_point_position[1]['position'][1]
                place_pose.position.z = place_point_position[1]['position'][2]


            # A list of dictionaries for output to yaml format
            # print("sending for yaml dict", test_scene_num, arm_name, object_name, pick_pose, place_pose )
            yaml_dict =  make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose )
            yaml_list.append(yaml_dict)
            # print("yaml dictionary given back", yaml_dict)

    
    ## Output your request parameters into output yaml file
    send_to_yaml(filename , yaml_list)

    # Wait for 'pick_place_routine' service to come up
    rospy.wait_for_service('pick_place_routine')

    # try:
    #     pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

    #     # TODO: Insert your message variables to be sent as a service request
    #     resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

    #     print ("Response: ",resp.success)

    # except rospy.ServiceException, e:
    #     print "Service call failed: %s"%e




if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node("clustering", anonymous = True)

    # TODO: Create Subscribers
    rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size = 1)

    # TODO: Create Publishers

    ## RANSAC publishing
    pcl_objects_pub = rospy.Publisher('/pcl_objects', PointCloud2, queue_size = 1)
    pcl_table_pub = rospy.Publisher('/pcl_table', PointCloud2, queue_size = 1)
    
    pcl_cluster_pub = rospy.Publisher('/pcl_cluster', PointCloud2, queue_size = 1)  
    
    object_markers_pub = rospy.Publisher('/object_markers', Marker, queue_size = 1)
    detected_objects_pub = rospy.Publisher('/detected_objects', DetectedObjectsArray, queue_size = 1)

    statistical_pub = rospy.Publisher('/statistical_downsampled', PointCloud2, queue_size = 1)
    vowel_pub = rospy.Publisher('/vowel_downsampled', PointCloud2, queue_size = 1)
    passthrough_pub = rospy.Publisher('/passthrough_downsampled', PointCloud2, queue_size = 1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()

