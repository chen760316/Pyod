import numpy as np
import matplotlib.pyplot as plt

# 提供的数据
algorithms = ['DeepSVDD', 'LOF', 'DAGMM', 'KNN', 'LODA+', 'CBLOF',
              'HBOS', 'COF', 'SOD', 'IForest+', 'COPOD', 'OCSVM', 'ECOD', 'PCA']
p_values = [61.25, 93.66, 79.39, 91.14, 84.12, 89.45, 85.40,
             90.77, 86.52, 88.67, 86.00, 86.78, 85.51, 87.04]

# 计算关键差异（CD）
# 这里我们将 p 值转换为 CD 值（你可以根据需要修改计算方式）
cd_values = np.array(p_values) / max(p_values)  # 归一化处理

# 绘制 CD 图
plt.figure(figsize=(12, 8))  # 调整图形大小

# 竖线表示每个算法的关键差异
for i, (algo, cd) in enumerate(zip(algorithms, cd_values)):
    plt.plot([i, i], [0, cd], 'k-', lw=2)  # 竖线宽度
    plt.text(i, cd + 0.05, algo, ha='center', va='bottom', fontsize=10)  # 调整文本位置和样式

# 设置 x 轴刻度和标签
plt.xticks(range(len(algorithms)), algorithms, rotation=45, ha='right')  # 旋转算法名称以便更好地显示

plt.title('Critical Difference Diagram', fontsize=16)  # 标题和字体大小
plt.xlabel('Algorithms', fontsize=14)  # x轴标签和字体大小
plt.ylabel('Normalized Critical Difference', fontsize=14)  # y轴标签和字体大小
plt.ylim(-0.1, 1.1)  # y轴范围
plt.axhline(0, color='black', lw=0.5)  # 添加 x 轴网格线
plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加 y 轴网格线，调整样式
plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域

plt.show()