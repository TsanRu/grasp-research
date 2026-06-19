import pandas as pd
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sklearn.cluster import KMeans
import numpy as np

def publish_handover_area_with_kmeans_and_spheres(df, n_clusters=3, topic='handover_area'):
    pub = rospy.Publisher(topic, Marker, queue_size=10)
    rospy.init_node('handover_area_viz', anonymous=True)

    # 綠色點雲（所有可達點）
    marker = Marker()
    marker.header.frame_id = "world"
    marker.ns = "reachable_points"
    marker.id = 0
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.scale.x = 0.01
    marker.scale.y = 0.01
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.points = []
    for idx, row in df.iterrows():
        pt = Point()
        pt.x = row['x']
        pt.y = row['y']
        pt.z = row['z']
        marker.points.append(pt)

    # KMeans 聚類
    X = df[['x', 'y', 'z']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # 紅色聚類中心
    center_marker = Marker()
    center_marker.header.frame_id = "world"
    center_marker.ns = "cluster_centers"
    center_marker.id = 1
    center_marker.type = Marker.POINTS
    center_marker.action = Marker.ADD
    center_marker.scale.x = 0.03
    center_marker.scale.y = 0.03
    center_marker.color.a = 1.0
    center_marker.color.r = 1.0
    center_marker.color.g = 0.0
    center_marker.color.b = 0.0
    center_marker.points = []
    for c in centers:
        pt = Point()
        pt.x, pt.y, pt.z = c
        print(f"center: ({pt.x}, {pt.y}, {pt.z}")
        center_marker.points.append(pt)

    # 每群畫一個半透明球體（顯示範圍）
    sphere_markers = []
    for i in range(n_clusters):
        cluster_points = X[labels == i]
        center = centers[i]
        distances = np.linalg.norm(cluster_points - center, axis=1)
        max_radius = np.max(distances)
        print(f"Radius: {max_radius}")
        sphere = Marker()
        sphere.header.frame_id = "world"
        sphere.ns = "cluster_spheres"
        sphere.id = 100 + i
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = center[0]
        sphere.pose.position.y = center[1]
        sphere.pose.position.z = center[2]
        sphere.scale.x = max_radius * 2
        sphere.scale.y = max_radius * 2
        sphere.scale.z = max_radius * 2
        sphere.color.a = 0.15  # 半透明
        sphere.color.r = 1.0
        sphere.color.g = 0.0
        sphere.color.b = 0.0
        sphere_markers.append(sphere)

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        pub.publish(marker)
        pub.publish(center_marker)
        for s in sphere_markers:
            pub.publish(s)
        rate.sleep()

if __name__ == "__main__":
    df = pd.read_csv('both_arms_reachable.csv')
    publish_handover_area_with_kmeans_and_spheres(df, n_clusters=3)
