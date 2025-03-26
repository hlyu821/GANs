from utils.load_data import LoadData_ml, LoadData_ml_reg

###处理图片，标签，划分数据集为CSV文件

# 设置输入和输出路径
input_path_class = 'dataset\data'  
output_path_class = 'dataset\label'

input_path_reg = 'dataset\data'  
output_path_reg = 'dataset\label_reg'

# 调用 LoadData 函数
LoadData_ml(input_path_class, output_path_class)

LoadData_ml_reg(input_path_reg, output_path_reg)


