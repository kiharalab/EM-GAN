from torch import mean, nn

class GeneratorLoss(nn.Module):
    def __init__(self, ltype):
        super(GeneratorLoss, self).__init__()
        self.ltype = ltype
        if ltype == "mse":
            self.loss = nn.MSELoss()
        elif ltype == "bce":
            print("here")
            self.loss = nn.BCEWithLogitsLoss()
        elif ltype == "comb":
            self.mloss = nn.MSELoss()
            self.bloss = nn.BCEWithLogitsLoss()

        # self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = mean(1 - out_labels)
        # Image Loss
        if self.ltype == "comb":
            image_loss = self.mloss(out_images, target_images) + 0.1*self.bloss(out_images, target_images)
        else:
            image_loss = self.loss(out_images, target_images)

        # image_loss = self.bce_loss(out_images, target_images)
        # TV Loss
#         return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss
        return image_loss + 0.001 * adversarial_loss
