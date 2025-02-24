import torch
import torch.nn as nn
from random import sample


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, r=16384, m=0.999, T=0.1, mlp=False):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature
        mlp: whether to use mlp projection
        """
        super(MoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.r % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.r  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, im_q, im_k=None, is_eval=False, cluster_result=None, index=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """

        if is_eval:
            k = self.encoder_k(im_q)
            k = nn.functional.normalize(k, dim=1)
            return k

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)


        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        # prototypical contrast
        if cluster_result is not None:
            proto_labels = []
            proto_logits = []
            for n, (im2cluster, prototypes, density) in enumerate(
                    zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])):
                # get positive prototypes
                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]

                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max() + 1)]
                # print('len of pos ids:')
                # print(len(set(pos_proto_id.tolist())))
                neg_proto_id = set(all_proto_id) - set(pos_proto_id.tolist())
                # print("all:")
                # print(neg_proto_id)
                # print('len of neg ids:')
                # print(len(neg_proto_id))
                # neg_proto_id = sample(neg_proto_id,self.r) #sample r negative prototypes
                # print('sample:')
                # print(neg_proto_id)
                neg_proto_id = list(neg_proto_id)
                neg_prototypes = prototypes[neg_proto_id]

                proto_selected = torch.cat([pos_prototypes, neg_prototypes], dim=0)

                # compute prototypical logits
                logits_proto = torch.mm(q, proto_selected.t())

                # targets for prototype assignment
                labels_proto = torch.linspace(0, q.size(0) - 1, steps=q.size(0)).long().cuda()

                # scaling temperatures for the selected prototypes
                temp_proto = density[torch.cat([pos_proto_id, torch.LongTensor(neg_proto_id).cuda()], dim=0)]
                logits_proto /= temp_proto

                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)
            return logits, labels, proto_logits, proto_labels
        else:
            return logits, labels, None, None

class ShapeMoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, r=16384, m=0.999, T=0.1, mlp=False):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature
        mlp: whether to use mlp projection
        """
        super(ShapeMoCo, self).__init__()

        self.r = r
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, r))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("shape_queue", torch.randn(dim, r))
        self.shape_queue = nn.functional.normalize(self.shape_queue, dim=0)

        self.register_buffer("shape_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, shape_keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        s_ptr = int(self.shape_queue_ptr)
        assert self.r % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.r  # move pointer

        self.queue_ptr[0] = ptr

        self.shape_queue[:, s_ptr:s_ptr + batch_size] = shape_keys.T
        s_ptr = (s_ptr + batch_size) % self.r  # move pointer

        self.queue_ptr[0] = ptr
        self.shape_queue_ptr[0] = s_ptr


    def forward(self, im_q, im_k=None, sp_k=None, is_eval=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            is_eval: return momentum embeddings (used for clustering)
            cluster_result: cluster assignments, centroids, and density
            index: indices for training samples
        Output:
            logits, targets, proto_logits, proto_targets
        """

        if is_eval:
            k = self.encoder_k(im_q)
            k = nn.functional.normalize(k, dim=1)
            return k

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            ks = self.encoder_k(sp_k)  # keys: NxC
            ks = nn.functional.normalize(ks, dim=1)


        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_pos_s = torch.einsum('nc,nc->n', [q, ks]).unsqueeze(-1)
        # negative logits: Nxr
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        l_neg_s = torch.einsum('nc,ck->nk', [q, self.shape_queue.clone().detach()])

        # logits: Nx(1+r)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_s = torch.cat([l_pos_s, l_neg_s], dim=1)

        # apply temperature
        logits /= self.T
        logits_s /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        labels_s = torch.zeros(logits_s.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k, ks)


        return logits, logits_s, labels, labels_s