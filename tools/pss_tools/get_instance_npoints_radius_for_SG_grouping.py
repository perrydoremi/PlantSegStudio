import numpy as np
import os
import matplotlib.pyplot as plt

def load_point_cloud(filepath):
    """Load point cloud from txt file"""
    return np.loadtxt(filepath)

def analyze_instances(points):
    """Analyze all instances except id 0"""
    instance_info = {}
    instances = np.unique(points[:,-1])
    
    for instance_id in instances:
        if instance_id != 0:
            instance_mask = points[:,-1] == instance_id
            instance_points = points[instance_mask]
            xyz = instance_points[:,:3]
            
            sphere_center = np.mean(xyz, axis=0) 
            radius = np.max(np.linalg.norm(xyz - sphere_center, axis=1))
            
            instance_info[int(instance_id)] = {
                'semantic_id': int(instance_points[0,-2]),
                'num_points': len(instance_points),
                'sphere_radius': radius
            }
            
    return instance_info

def process_dataset_directory(base_dirs):
    """Process all point clouds in multiple directories and calculate class averages"""
    all_results = []
    
    for base_dir in base_dirs:
        dataset_results = []
        # Dictionary to store points per class
        class_points = {}
        
        for filename in os.listdir(base_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(base_dir, filename)
                points = load_point_cloud(filepath)
                results = analyze_instances(points)
                
                if results:
                    dataset_results.append({
                        'file': filename,
                        'instances': results
                    })
                    
                    # Group instances by semantic class and collect point counts
                    for instance_data in results.values():
                        semantic_id = instance_data['semantic_id']
                        # Skip class 0
                        if semantic_id != 0:
                            if semantic_id not in class_points:
                                class_points[semantic_id] = []
                            class_points[semantic_id].append(instance_data['num_points'])
        
        # Calculate class averages
        class_averages = {}
        for class_id, points_list in class_points.items():
            if points_list:
                class_averages[class_id] = np.mean(points_list)
        
        all_radii = []
        for result in dataset_results:
            all_radii.extend([info['sphere_radius'] for info in result['instances'].values()])
            
        summary = {
            'dataset_dir': base_dir,
            'num_clouds': len(dataset_results),
            'mean_radius': np.mean(all_radii) if all_radii else 0,
            'std_radius': np.std(all_radii) if all_radii else 0,
            'cloud_results': dataset_results,
            'class_averages': class_averages
        }
        all_results.append(summary)
        
        print(f"\nResults for {base_dir}")
        print(f"Number of point clouds: {len(dataset_results)}")
        if all_radii:
            print(f"Mean sphere radius: {np.mean(all_radii):.3f}")
            print(f"Std sphere radius: {np.std(all_radii):.3f}")
        
        # Print average points per class
        print("\nAverage number of points per class:")
        for class_id, avg_points in sorted(class_averages.items()):
            print(f"Class {class_id}: {avg_points:.2f}")
        
        # For datasets with 3 classes, output averages for class 1 and 2 specifically
        num_classes = len(class_averages)
        if num_classes == 3 and 1 in class_averages and 2 in class_averages:
            print("\nFor 3-class dataset, specific averages:")
            print(f"Class 1: {class_averages[1]:.2f}")
            print(f"Class 2: {class_averages[2]:.2f}")
    
    return all_results

if __name__ == "__main__":
    dataset_dirs = [
        'data/plant/COS/COS',
        'data/plant/HR3DAllThree/HR3DAllThree',
        'data/plant/SYAUMaize/SYAUMaize', 
        'data/plant/Pheno4DAllRS/Pheno4DAllRS',
        'data/plant/SoyBeanMVSRS/SoyBeanMVSRS'
    ]
    
    results = process_dataset_directory(dataset_dirs)