import torch
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from Dataset.dataset import OliveCropsDataset
from Models.convolutional_models import Net
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm


crop_size = (120, 120)
dataset_name = f'Data{crop_size[0]}x{crop_size[1]}'
dataset_csv_path = 'Dataset/'+dataset_name+'/DatasetCrop.csv'
dataset_images_path = 'Dataset/'+dataset_name+'/images'

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])
validationset = OliveCropsDataset(dataset_csv_path, root_dir=dataset_images_path, learn_set='validation', transform=transform)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=len(validationset), shuffle=True, num_workers=0)
classes = ('superintensivo', 'intensivo', 'tradicional')

img, _ = validationset[0]

model = Net(input_size=tuple(img[0].shape), output_size=len(classes))
model.load_state_dict(torch.load(f'./Models/trained_models/classifierData{crop_size[0]}x{crop_size[1]}.pth'))
model.eval()

with torch.no_grad():

	# Sample the dataloader #
	new_data, true_labels = iter(validationloader).next()
	# Predict #
	predicted_values = model(new_data).cpu()
	_, predicted_classes = torch.max(predicted_values, 1)

	accuracy = accuracy_score(y_true=true_labels, y_pred=predicted_classes)
	macro_precision = precision_score(y_true=true_labels, y_pred=predicted_classes, average='macro')
	roc_auc_ovo = roc_auc_score(y_true=true_labels, y_score=softmax(predicted_values, dim=1),multi_class='ovo')
	roc_auc_ovr = roc_auc_score(y_true=true_labels, y_score=softmax(predicted_values, dim=1),multi_class='ovr')
	f1 = f1_score(y_true=true_labels, y_pred=predicted_classes, average='macro')
	confusion = confusion_matrix(y_true=true_labels, y_pred=predicted_classes)

	print(f"Precision: {macro_precision}, Accuracy: {accuracy}, roc_auc_ovo score: {roc_auc_ovo}, roc_auc_ovr score: {roc_auc_ovr}")

	disp = ConfusionMatrixDisplay(confusion, display_labels=classes)

	disp.plot(cmap='Blues')

	plt.show()
