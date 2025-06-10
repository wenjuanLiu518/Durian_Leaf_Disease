import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from scipy.spatial.distance import cdist, pdist

def compute_kid(real_features, fake_features, n_subsets=5, subset_size=50):
    """
    稳健的KID计算实现
    参数：
        real_features: 真实样本特征矩阵 (n_samples, n_features)
        fake_features: 生成样本特征矩阵 (n_samples, n_features)
        n_subsets: 子集采样次数
        subset_size: 每个子集大小
    """
    # 数据预处理：L2归一化
    real_norm = real_features / np.linalg.norm(real_features, axis=1, keepdims=True)
    fake_norm = fake_features / np.linalg.norm(fake_features, axis=1, keepdims=True)
    # real_norm = real_features
    # fake_norm = fake_features
    
    # 自动选择gamma范围（基于数据统计）
    pairwise_dist = np.median(cdist(real_norm, real_norm, 'sqeuclidean'))
    gamma_auto = 1.0 / (2 * pairwise_dist)
    gamma_values = [gamma_auto * (0.5 ** i) for i in range(-2, 3)]  # 生成5个自适应gamma
    print("gamma_values===>>>", gamma_values)
    
    kid_scores = []
    
    for _ in range(n_subsets):
        # 随机子集采样
        real_subset = real_norm[np.random.choice(len(real_norm), min(len(real_norm), subset_size), replace=False)]
        fake_subset = fake_norm[np.random.choice(len(fake_norm), min(len(fake_norm), subset_size), replace=False)]
        
        # 多核MMD计算
        mmd_sq = 0
        for gamma in gamma_values:
            # 计算RBF核矩阵
            k_real_real = rbf_kernel(real_subset, real_subset, gamma=gamma)
            k_fake_fake = rbf_kernel(fake_subset, fake_subset, gamma=gamma)
            k_real_fake = rbf_kernel(real_subset, fake_subset, gamma=gamma)
            
            # MMD²计算（数值稳定版本）
            term1 = (k_real_real.sum() - np.trace(k_real_real)) / (subset_size * (subset_size - 1))
            term2 = (k_fake_fake.sum() - np.trace(k_fake_fake)) / (subset_size * (subset_size - 1))
            term3 = k_real_fake.mean() * 2
            mmd_sq += max(term1 + term2 - term3, 0)  # 保证非负
            
        # 多核平均，开平方为NMD
        # kid_scores.append(np.sqrt(mmd_sq / len(gamma_values)))
        # KID = NMD的平方
        kid_scores.append(mmd_sq / len(gamma_values))
    
    return np.mean(kid_scores), np.std(kid_scores)


if __name__ == "__main__":
    # 示例用法
    real_features = np.random.randn(100, 768)  # 真实样本
    fake_features = np.random.randn(100, 768)  # 生成样本

    kid_mean, kid_std = compute_kid(real_features, fake_features)
    print(f"KID值: {kid_mean:.4f} ± {kid_std:.4f}")