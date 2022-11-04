import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        resnet = models.resnet34(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = True
        layers = list(resnet.children())[:8]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 1))
        self.bb_regression = nn.Sequential(nn.Conv2d(kernel_size=(1, 1), in_channels=512, out_channels=6), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        # x.shape = _, 512, 13, 19
        x_class = nn.AdaptiveAvgPool2d((1, 1))(x)
        x_class = x_class.view(x.shape[0], -1)
        x_class = self.classifier(x_class)
        x_bb = self.bb_regression(x)  # x_bb.shape = _, 6, 13, 19

        return x_class, x_bb

    def initialize_weights(self):
        pass


class ModelLoss:

    def __init__(self, weights: [float], width: float, height: float):
        self.weights = weights
        self.epsilon = 1e-4
        self.width = width
        self.height = height

    def losses(self, out_classes, target_classes, out_bbs, target_bbs):
        # out_bbs.shape = _, 6, 13, 19
        # 6 => p_x0, x01, x02, p_x1, x11, x12
        loss_presence = F.binary_cross_entropy_with_logits(out_classes, target_classes.unsqueeze(1), reduction="sum")
        # Reshape
        B, N, H, W = out_bbs.shape
        HxW = H * W
        preds = out_bbs.reshape(B, N, HxW)
        preds = preds.transpose(1, 2).contiguous()  # B, H x W, N
        oneobj_hat = self.get_one_obj(preds)
        preds = preds.reshape(B * HxW, N)
        preds = torch.cat([preds[:, 1:3], preds[:, 4:]])
        oneobj_target = self.get_one_obj_target(target_bbs, height=H, width=W)
        loss_cell_presence = F.binary_cross_entropy(oneobj_hat, oneobj_target, reduction="sum")
        reshaped_target_boxes = self.reshape_target_boxes(target_bbs, height=H, width=W)
        loss_distance_to_corners = (((preds - reshaped_target_boxes) ** 2).sum(dim=1) * oneobj_hat).sum()
        return loss_presence, loss_cell_presence, loss_distance_to_corners

    def aggregate_losses(self, losses):
        my_loss = 0
        for i in range(len(losses)):
            my_loss += losses[i] * self.weights[i]
        return my_loss

    def loss(self, out_classes, target_classes, out_bbs, target_bbs):
        curr_losses = self.losses(out_classes, target_classes, out_bbs, target_bbs)
        return self.aggregate_losses(curr_losses)

    def reshape_output(self, predictions):
        """
        Transforms the output of shape (B, F, H, W) into (B, H * W * F)
        :param predictions: Output of the network
        :return: Reshaped output
        """
        B, F, H, W = predictions.shape
        preds = predictions.reshape(B, F, H * W)
        preds = preds.transpose(1, 2).contiguous()  # B, H x W, F
        preds = preds.reshape(B, H * W, 2, int(F / 2))
        return preds

    def get_one_obj(self, preds):
        """
        preds.shape = # B, H x W, N
        :return tensor of shape B * HxW * 2 with one where the corner was predicted
        """
        return torch.cat([preds[:, :, 0].flatten(), preds[:, :, 3].flatten()])

    def get_one_obj_target(self, target_bbs, height: int, width: int):
        """
        Transforme les target boxes en un grand one-hot vector de longueur B x H x W x 2.
        Il y a un 1 si la case est censée contenir un coin. Le coin en haut à gauche est contenu dans les B x H x W
        premières cellules. Le coin en bas à droite est contenu dans les B x H x W dernières cellules (d'où le x2 dans
        les dimensions).
        :param target_bbs: tensor de taille B x 4
        :param height: le nombde de cellules par image en hauteur
        :param width: le nombde de cellules par image en largeur
        :return: vecteur de taille B x H x W x 2
        """
        corners = self.get_cell_with_corners(target_bbs, height=height, width=width)
        HxW = height * width
        B = target_bbs.shape[0]
        size_coord = B * HxW
        oneobj = torch.zeros(size_coord * 2, requires_grad=False, device=target_bbs.device)
        for i, val in enumerate(corners[:, 0]):
            oneobj[i * HxW + int(val)] = 1.
        for i, val in enumerate(corners[:, 1]):
            oneobj[i * HxW + int(val) + size_coord] = 1.
        return oneobj

    def reshape_target_boxes(self, target_bbs, height, width):
        """
        Transforme les target boxes (B, 4) en torch.tensor (B x H x W x 2, 2)
        Les B x H x W premières entrées contiennent les coordonnées du coin gauche supérieur et les B x H x W
        dernières les coordonnées du coin droit inférieur.
        Les coordonnées sont relatives au coin haut gauche de la cellule.
        :param target_bbs: Targets boxes initiales
        :param height: Nombre de cellules par hauteur
        :param width: Nombre de cellules par largeur
        :return:
        """
        # TODO peut-être douteux
        B, _ = target_bbs.shape
        HxW = height * width
        targets = torch.cat([coord.expand(HxW, -1) for coord in torch.cat([target_bbs[:, 0:2], target_bbs[:, 2:]])])
        template_origin = torch.zeros((HxW, 2), requires_grad=False, device=target_bbs.device)
        idx = 0
        for i in range(height):
            ii = i / height
            for j in range(width):
                template_origin[idx] = torch.tensor([ii, j / width], requires_grad=False, device=target_bbs.device)
                idx += 1
        template_origin = template_origin.repeat(B * 2, 1)
        targets = (targets - template_origin) * torch.tensor([height, width], requires_grad=False,
                                                             device=targets.device)
        return targets

    def get_cell_with_corners(self, target_bbs, height, width):
        dims = torch.tensor([height, width, height, width], requires_grad=False, device=target_bbs.device)
        new_tbs = torch.floor(target_bbs * dims)
        corners = torch.cat([
            (new_tbs[:, 0] * width + new_tbs[:, 1]).unsqueeze(0),
            (new_tbs[:, 2] * width + new_tbs[:, 3]).unsqueeze(0)
        ], dim=0).transpose(0, 1)
        return corners


def get_bb_from_bouding_boxes(predicted, height: int, width: int):
    """
    :param predicted: Tensor of shape (B, 6, H, W)
    :return: Tensor shape (B, 4)
    """
    B, N, H, W = predicted.shape
    HxW = H * W
    cell_height = height / H
    cell_width = width / W
    preds = predicted.reshape(B, N, HxW)
    preds = preds.transpose(1, 2).contiguous()  # B, H x W, N
    _, indices_y = torch.max(preds[:, :, 0], dim=1)
    _, indices_x = torch.max(preds[:, :, 3], dim=1)
    origins = torch.cat(
        [
            torch.cat([torch.tensor([torch.floor(val / W), val % W]) + preds[i, val, 1:3] for i, val in
                       enumerate(indices_y)]).unsqueeze(0),
            torch.cat([torch.tensor([torch.floor(val / W), val % W]) + preds[i, val, 4:] for i, val in
                       enumerate(indices_x)]).unsqueeze(0)
        ], dim=1
    )
    return origins * torch.tensor([cell_height, cell_width, cell_height, cell_width])
