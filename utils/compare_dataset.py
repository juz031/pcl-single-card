import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import scipy.linalg
from matplotlib import pyplot as plt

def cat_matrix(dataset_dir):
    img_matrix = []
    cats = os.listdir(dataset_dir)
    for cat in tqdm(cats):
        cat_dir = os.path.join(dataset_dir, cat)
        for img_name in os.listdir(cat_dir):
            img_path = os.path.join(cat_dir, img_name)
            img = Image.open(img_path)
            img = np.array(img.convert('RGB').resize((64, 64)).getdata())
            img_vector = img.flatten()
            img_matrix.append(img_vector)
    
    img_matrix = np.array(img_matrix)
    print(img_matrix.shape)
    
    return img_matrix


def create_matrix(img_dir):
    img_matrix = []
    for img_name in sorted(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path)
            img = np.array(img.convert('RGB').resize((64, 64)).getdata())
            img_vector = img.flatten()
            img_matrix.append(img_vector)
    
    img_matrix = np.array(img_matrix)
    
    return img_matrix

def multivariate_gaussian(data):
    miu = np.mean(data, axis=0, keepdims=True)
    M = data - miu
    cov = M.T @ M / data.shape[0]

    return miu, cov


# def kl_mvn(m_to, S_to, m_fr, S_fr):
#     """Calculate `KL(to||fr)`, where `to` and `fr` are pairs of means and covariance matrices"""    
#     d = m_fr - m_to
    
#     c, lower = scipy.linalg.cho_factor(S_fr)
#     def solve(B):
#         return scipy.linalg.cho_solve((c, lower), B)
    
#     def logdet(S):
#         return np.linalg.slogdet(S)[1]

#     term1 = np.trace(solve(S_to))
#     term2 = logdet(S_fr) - logdet(S_to)
#     term3 = d.T @ solve(d)
#     return (term1 + term2 + term3 - len(d))/2.


def kl_mvn(m0, S0, m1, S1):
    """
    https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
    The following function computes the KL-Divergence between any two 
    multivariate normal distributions 
    (no need for the covariance matrices to be diagonal)
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    - accepts stacks of means, but only one S0 and S1
    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    # 'diagonal' is [1, 2, 3, 4]
    tf.diag(diagonal) ==> [[1, 0, 0, 0]
                          [0, 2, 0, 0]
                          [0, 0, 3, 0]
                          [0, 0, 0, 4]]
    # See wikipedia on KL divergence special case.              
    #KL = 0.5 * tf.reduce_sum(1 + t_log_var - K.square(t_mean) - K.exp(t_log_var), axis=1)   
                if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))                               
    """
    # store inv diag covariance of S1 and diff between means
    m0, m1 = m0.T, m1.T
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N).squeeze()


if __name__ == '__main__':
    train_dir = "/user_data/junruz/IN-shape-10-73/set_1/split_0/train"
    val_dir = "/user_data/junruz/IN-shape-10-73/set_1/split_0/train"
    # train_dir = '/user_data/junruz/imagenet_shape_10/train_1'
    # val_dir = '/user_data/junruz/imagenet_shape_10/val_1'

    cats = sorted(os.listdir(train_dir))
    KLs = []
    for cat in tqdm(cats):
        train_path = os.path.join(train_dir, cat)
        val_path = os.path.join(val_dir, cat)
        train_matrix = create_matrix(train_path) / 255.
        val_matrix = create_matrix(val_path) /255.

        n_tr = train_matrix.shape[0]
        n_val = val_matrix.shape[0]

        all_matrix = np.vstack((train_matrix, val_matrix))
        all_matrix -= np.mean(all_matrix, axis=0, keepdims=True)
        all_matrix = all_matrix.T
        U, s, v = np.linalg.svd(all_matrix)
        proj = U[:, :48]
        proj = proj.T

        all_matrix_lowd = (proj @ all_matrix).T

        train_matrix_lowd = all_matrix_lowd[:n_tr]
        val_matrix_lowd = all_matrix_lowd[n_tr:n_tr+n_val]

        miu_tr, cov_tr = multivariate_gaussian(train_matrix_lowd)
        miu_val, cov_val = multivariate_gaussian(val_matrix_lowd)

        kld_73 = kl_mvn(miu_tr, cov_tr, miu_val, cov_val)
        print(kld_73)
        KLs.append(kld_73)

    mean_KL = np.mean(KLs)
    print(f'mean KL: {mean_KL}')


