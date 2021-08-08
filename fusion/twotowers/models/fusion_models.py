import torch
import torch.nn as nn
from pytorch_tabnet import tab_network

class LateFusionModel(nn.Module):

    def __init__(
            self, 
            cfg,
            emr_model, 
            image_model, 
            num_classes=1, 
            aggregation_type='mean',
            freeze_img_model=False,
            freeze_emr_model=False):
        super().__init__()

        self.cfg = cfg
        self.emr_model = emr_model
        self.image_model = image_model
        self.aggregation_type = aggregation_type
        if self.aggregation_type == 'fcnn': 
            self.fc = nn.Linear(2, num_classes)

        # freeze backbone models
        if freeze_emr_model:
            for param in self.emr_model.parameters():
                param.requires_grad = False
        if freeze_img_model: 
            for param in self.image_model.parameters():
                param.requires_grad = False

    def forward(self, emr, image):
        # TODO: what to do with m loss in this case?
        # TODO: should we average on logit or prob?

        emr_logit, M_loss = self.emr_model(emr)
        image_logit = self.image_model(image)
        logits = torch.cat([emr_logit, image_logit], axis=1)

        if self.aggregation_type == 'mean':
            output = torch.mean(logits, axis=1)
        elif self.aggregation_type == 'fcnn': 
            output = self.fc(logits)
        else: 
            raise Exception(f'Aggregation type {self.aggregation_type} not supported')

        #emr_prob = torch.sigmoid(emr_logit)
        #image_prob = torch.sigmoid(image_logit)
        #output = torch.mean([emr_prob, image_prob])

        return output 


class JointFusionModel(nn.Module):

    def __init__(
            self, 
            cfg,
            emr_model, 
            image_model,
            num_classes=1, 
            use_batchnorm=False,
            num_hidden=0,
            dropout_prob=0.0,
            freeze_img_model=False,
            freeze_emr_model=False):
        super().__init__()

        self.cfg = cfg
        self.use_batchnorm = use_batchnorm
        self.num_hidden = num_hidden 
        self.dropout_prob = dropout_prob
        self.emr_model = emr_model
        self.image_model = image_model

        emr_out_dim = self.emr_model.n_d
        img_out_dim = self.image_model.n_kernels
        feat_dim = emr_out_dim + img_out_dim

        self.batchnorm = nn.BatchNorm1d(feat_dim)

        # output layers 
        fc_layers = []
        for _ in range(num_hidden):
            fc_layers.append(nn.Linear(feat_dim, feat_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=self.dropout_prob))
        fc_layers.append(nn.Linear(feat_dim, num_classes))
        self.fc_layers = nn.Sequential(*fc_layers)

        # freeze backbone models
        if freeze_emr_model:
            for param in self.emr_model.parameters():
                param.requires_grad = False
        if freeze_img_model: 
            for param in self.image_model.parameters():
                param.requires_grad = False

    def forward(self, emr, image):

        image_features = self.image_model.forward(image, get_features=True) 
        emr_features = self.emr_forward(emr) 
        joint_features = torch.cat((image_features, emr_features), 1)
        if self.use_batchnorm:
            joint_features = self.batchnorm(joint_features)
        output = self.fc_layers(joint_features)

        return output 

    def emr_forward(self, emr): 
        x = self.emr_model.embedder(emr)
        res = 0
        steps_output, M_loss = self.emr_model.tabnet.encoder(x)
        res = torch.sum(torch.stack(steps_output, dim=0), dim=0)
        return res

class EarlyFusionModel(nn.Module):
    
    def __init__(self, cfg, model, **kwargs): 
        super().__init__()

        self.cfg = cfg

        # modify input dimentions
        new_embed_dim = model.post_embed_dim + 128  # with image features
        self.model = model
        self.model.post_embed_dim = new_embed_dim
        self.model.tabnet = tab_network.TabNetNoEmbeddings(
            new_embed_dim,
            model.output_dim,
            model.n_d, 
            model.n_a,
            model.n_steps,
            model.gamma, 
            model.n_independent, 
            model.n_shared,
            model.epsilon, 
            model.virtual_batch_size, 
            self.cfg.model.emr.momentum, 
            model.mask_type 
        )

    def forward(self, emr, image):

        emr_emb = self.model.embedder(emr)
        x = torch.cat([emr_emb, image], dim=1)
        logit, M_loss = self.model.tabnet(x)

        return logit, M_loss

