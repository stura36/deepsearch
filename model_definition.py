import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torchvision.models import resnet50, ResNet50_Weights
import lightning.pytorch as pl
import torch
from deepsearch.metrics import calc_r_at_k


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
device = "cpu"

MAX_LEN = 39

pretrained_text_model = AutoModel.from_pretrained(
    "sentence-transformers/all-mpnet-base-v2"
)

weights = ResNet50_Weights.IMAGENET1K_V1
pretrained_viz_model = resnet50(weights=weights)
preprocess = weights.transforms()


# Function for pooling transformer output
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class VizModel(pl.LightningModule):
    def __init__(self, pretrained_model, emb_size):
        super().__init__()
        for param in pretrained_model.parameters():
            param.requires_grad = False

        in_features = pretrained_model.fc.in_features
        out_features = 100

        pretrained_model.fc = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(in_features, emb_size)
        )
        pretrained_model.fc.requires_grad = True

        self.pre_mdl = pretrained_model

    def forward(self, x):
        x = self.pre_mdl(x)
        return x


class TextModel(pl.LightningModule):
    def __init__(self, pretrained_model, emb_size, pooler=mean_pooling):
        super(TextModel, self).__init__()
        for param in pretrained_model.parameters():
            param.requires_grad = False

        self.pre_mdl = pretrained_model
        self.pooler = pooler
        self.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, emb_size))

        self.fc.requires_grad = True
        # pretrained_model.pooler.requires_grad = True

    def forward(self, input_ids, attention_mask):
        pre_model_out = self.pre_mdl(input_ids, attention_mask)
        embeddings = self.pooler(pre_model_out, attention_mask)
        x = self.fc(embeddings)
        return x


class ModelComposition(pl.LightningModule):
    def __init__(
        self,
        pretrained_txt_model,
        pretrained_viz_model,
        pooler,
        criterion,
        emb_size,
        lr,
        max_lr,
        tokenizer,
    ):
        super(ModelComposition, self).__init__()

        self.viz_model = VizModel(pretrained_viz_model, emb_size)
        self.txt_model = TextModel(pretrained_txt_model, emb_size, pooler)
        self.criterion = criterion 
        self.lr = lr
        self.max_lr = max_lr

        self.preprocess_img = preprocess
        self.preprocess_txt = tokenizer 

        self.save_hyperparameters()

        self.viz_output_val = []
        self.txt_output_val = []

    def forward(self, imgs, input_ids, attention_mask):
        viz_output = self.viz_model(imgs)
        txt_output = self.txt_model(input_ids, attention_mask)

        return viz_output, txt_output

    def training_step(self, batch, batch_idx):
        imgs, input_ids, attention_mask = batch
        viz_output, txt_output = self(imgs, input_ids, attention_mask)
        loss = self.criterion(viz_output, txt_output)

        r1img, r5img, r10img, r1txt, r5txt, r10txt = calc_r_at_k(viz_output, txt_output)

        self.log("r1img", r1img)
        self.log("r5img", r5img)
        self.log("r10img", r10img)

        self.log("r1txt", r1txt)
        self.log("r5txt", r5txt)
        self.log("r10txt", r10txt)

        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, input_ids, attention_mask = batch
        viz_output, txt_output = self(imgs, input_ids, attention_mask)

        self.viz_output_val.append(viz_output)
        self.txt_output_val.append(txt_output)
        # turn to tensor after epoch and calculate the logs

        loss = self.criterion(viz_output, txt_output)

        self.log("loss_val", loss)
        return loss

    def on_validation_epoch_end(self):
        viz_output = torch.concat(self.viz_output_val)
        txt_output = torch.concat(self.txt_output_val)

        r1img, r5img, r10img, r1txt, r5txt, r10txt = calc_r_at_k(viz_output, txt_output)

        self.log("r1img_val", r1img)
        self.log("r5img_val", r5img)
        self.log("r10img_val", r10img)

        self.log("r1txt_val", r1txt)
        self.log("r5txt_val", r5txt)
        self.log("r10txt_val", r10txt)

        self.viz_output_val.clear()
        self.txt_output_val.clear()

        return

    def inference_image(self, image):
        image_preproc = self.preprocess_img(image)
        image_preproc = torch.unsqueeze(image_preproc, 0)
        image_emb = self.viz_model(image_preproc)

        return image_emb

    def inference_txt(self, txt):
        txt_preproc = self.preprocess_txt(txt, return_tensors="pt")

        txt_emb = self.txt_model(
            txt_preproc["input_ids"], txt_preproc["attention_mask"]
        )

        return txt_emb

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer  # , [scheduler]




retrieval_model = ModelComposition(
    pretrained_text_model,
    pretrained_viz_model,
    mean_pooling,
    None,
    1000,
    None,
    None,
    tokenizer,
)
