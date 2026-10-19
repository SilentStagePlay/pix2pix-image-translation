import Augmentor


p = Augmentor.Pipeline('/media/sgu/软件/PythonProjects/gujiarui/image_in_paper/aug_image')
p.ground_truth('/media/sgu/软件/PythonProjects/gujiarui/image_in_paper/aug_mask')
p.rotate90(probability=0.5)
p.rotate180(probability=0.5)
p.rotate270(probability=0.5)
p.rotate(probability=0.5, max_left_rotation=25, max_right_rotation=25)
p.skew_tilt(probability=0.5, magnitude=1)
p.skew_corner(probability=0.5, magnitude=1)
p.random_distortion(probability=0.5, grid_width=6, grid_height=6, magnitude=50)
p.shear(probability=0.5, max_shear_left=25, max_shear_right=25)
p.flip_top_bottom(probability=0.5)
p.zoom_random(probability=0.5, percentage_area=0.8)
p.sample(10)
